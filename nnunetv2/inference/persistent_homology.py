''' Code based on the work in 
https://github.com/nick-byrne/topological-losses and https://github.com/MIC-DKFZ/nnUNet 
based on 
N. Byrne, J. R. Clough, I. Valverde, G. Montana and A. P. King, 
"A persistent homology-based topological loss for CNN-based multi-class segmentation of CMR," 
in IEEE Transactions on Medical Imaging, 2022, doi: 10.1109/TMI.2022.3203309. '''


import torch
import torch.nn.functional as F
from torch._dynamo import OptimizedModule
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.inference.data_iterators import PreprocessAdapterFromNpy
from nnunetv2.inference.export_prediction import export_prediction_from_logits, \
    convert_predicted_logits_to_segmentation_with_correct_shape
import numpy as np
import copy
import cripser as crip
import tcripser as trip
import matplotlib.pyplot as plt
import os
# from multiprocessing import Pool
from typing import Union
from nnunetv2.utilities.helpers import empty_cache, dummy_context
from acvl_utils.cropping_and_padding.padding import pad_nd_image
import pickle
import pathlib


class nnUNetPredictorPH(nnUNetPredictor):
    
    def __init__(self,
                tile_step_size: float = 0.5,
                use_gaussian: bool = True,
                use_mirroring: bool = True,
                perform_everything_on_device: bool = True,
                device: torch.device = torch.device('cuda'),
                verbose: bool = False,
                verbose_preprocessing: bool = False,
                allow_tqdm: bool = True,
                priors: dict = None):
        
        nnUNetPredictor.__init__(self, tile_step_size, use_gaussian, use_mirroring, perform_everything_on_device, device, verbose, verbose_preprocessing, allow_tqdm)
        self.priors = priors

    def crip_wrapper(self, X, D):
        return crip.computePH(X, maxdim=D)

    def trip_wrapper(self, X, D):
        return trip.computePH(X, maxdim=D)

    def get_roi(X, thresh=0.01):
        true_points = torch.nonzero(X >= thresh)
        corner1 = true_points.min(dim=0)[0]
        corner2 = true_points.max(dim=0)[0]
        roi = [slice(None, None)] + [slice(c1, c2 + 1) for c1, c2 in zip(corner1, corner2)]
        return roi


    def checkIndexing(self, indexes, shape):
        shapeUpLimit = np.array(shape)[np.newaxis,:] - 1
        shapeDownLimit = np.array((0,0,0))[np.newaxis,:]
        idxsOut    = (indexes > shapeUpLimit).nonzero()
        indexes[idxsOut] = shapeUpLimit[0,idxsOut[1]]
        idxsOut    = (indexes < shapeDownLimit).nonzero()
        indexes[idxsOut] = shapeDownLimit[0,idxsOut[1]]


    def get_differentiable_barcode(self, tensor, barcode, shape):
        '''Makes the barcode returned by CubicalRipser differentiable using PyTorch.
        Note that the critical points of the CubicalRipser filtration reveal changes in sub-level set topology.
        
        Arguments:
            REQUIRED
            tensor  - PyTorch tensor w.r.t. which the barcode must be differentiable
            barcode - Barcode returned by using CubicalRipser to compute the PH of tensor.numpy()
            shape   - The shape of the tensor that was obtained from combo_arr in order to avoid bad indexing due to ph N constructor overpassing tensor size
        '''
        # Identify connected component of ininite persistence (the essential feature)
        inf = barcode[barcode[:, 2] == np.finfo(barcode.dtype).max]
        fin = barcode[barcode[:, 2] < np.finfo(barcode.dtype).max]

        # Get birth of infinite feature
        # with construction N sometimes the tensor is indexed out of range which causes 
        # IndexKernel.cu:91: block: [0,0,0], thread: [0,0,0] Assertion `index >= -sizes[i] && index < sizes[i] && "index out of bounds"` failed.
        # This is because the barcode gives indexes (xyz birth or death) out of range so we checked and adjusted to the boundaries. I saw this for the 
        # inf entity but I applied it to the others as well just in case
        indexes = inf[:, 3:3+tensor.ndim].astype(np.int64)
        self.checkIndexing(indexes, shape)
        inf_birth = tensor[tuple(indexes.T)]

        # Calculate lifetimes of finite features
        indexes = fin[:, 3:3+tensor.ndim].astype(np.int64)
        self.checkIndexing(indexes, shape)
        births = tensor[tuple(indexes.T)]
        indexes = fin[:, 6:6+tensor.ndim].astype(np.int64)
        self.checkIndexing(indexes, shape)
        deaths = tensor[tuple(indexes.T)]
        delta_p = (deaths - births)

        # Split finite features by dimension
        delta_p = [delta_p[fin[:, 0] == d] for d in range(tensor.ndim)]

        # Sort finite features by persistence
        delta_p = [torch.sort(d, descending=True)[0] for d in delta_p]

        return inf_birth, delta_p


    def multi_class_topological_post_processing(
        self, 
        inputs, 
        resPath,
        lr=1e-5, mse_lambda=1000,
        opt=torch.optim.Adam, num_its=100, construction='N', thresh=None, parallel=False,
        ):
        '''Performs topological post-processing.
        '''
        # Get image properties
        spatial_xyz = list(inputs.shape[1:])

        # Get raw prediction
        self.network.eval()
        with torch.no_grad():
            prediction = self.predict_sliding_window_return_logits(inputs)
            prediction = torch.softmax(prediction, 0)

        #Determine the prior based on max class
        maxClassFound = prediction.argmax(dim=0).cpu().numpy().max()
        if maxClassFound==4:
            prior = self.priors['D8']
        elif maxClassFound==3:
            prior = self.priors['MX']
        elif maxClassFound <= 2:
            prior = self.priors['default']
        else:
            raise ValueError('Did you put the right image to segment?')

        # If appropriate, choose ROI for topological consideration
        if thresh:
            roi = self.get_roi(prediction[1:].sum(0).squeeze(), thresh)
            with open(os.path.join(resPath, "roi.pickle"), 'wb') as f:
                pickle.dump(roi, f)
        else:
            roi = [slice(None, None)] + [slice(None, None) for dim in range(len(spatial_xyz))]

        # Initialise topological model and optimiser - here in nnunet the model topo should be network, and it is not returned
        # it remains in self.network and then is used
        # model_topo = copy.deepcopy(self.network)
        # model_topo.train()
        self.network.train()
        optimiser = opt(self.network.parameters(), lr=lr)

        # Inspect prior and convert to tensor
        max_dims = [len(b) for b in prior.values()]
        prior = {torch.tensor(c): torch.tensor(b) for c, b in prior.items()}

        # Set mode of cubical complex construction
        ph = {'0': self.crip_wrapper, 'N': self.trip_wrapper}

        for it in range(num_its):

            # Reset gradients
            optimiser.zero_grad()

            # Get current prediction
            # outputs = torch.softmax(model_topo(inputs), 1).squeeze()
            outputs = self.predict_sliding_window_return_logits(inputs)
            outputs = torch.softmax(outputs, 0)
            outputs_roi = outputs[roi]

            # Build class/combination-wise (c-wise) image tensor for prior
            # Here we adjust this to do not have the c.T warning
            # as all the c or prior keys should be 1D then is not reason to do c.T
            # pytorch raises a warning
            tmp = []
            for c in prior.keys():
                if c.dim()== 1:
                    tmp.append(outputs_roi[c].sum(0))
                else:
                    raise ValueError("Wrong dim")
            combos = torch.stack(tmp)

            if it%25==0 or it==num_its-1:

                combosArr = combos.cpu().detach().numpy()
                nImgs = combosArr.shape[0]+1
                shown_slice = int(combosArr.shape[1]/2)
                ncols=3
                nrows = int(np.ceil(nImgs/ncols))
                _, a = plt.subplots(nrows, ncols)
                if a.ndim == 1 : a = a[np.newaxis,:]
                j=0
                i=0
                for nImg in range(nImgs-1):
                    a[j,i].imshow(combosArr[nImg,shown_slice,:,:], interpolation='none')
                    a[j,i].axis('off')
                    i+=1
                    if i % ncols == 0: 
                        j += 1
                        i = 0
                a[j,i].imshow(inputs[roi].cpu().numpy()[0,shown_slice,:,:], interpolation='none')
                plt.savefig(os.path.join(resPath, "it_{}.png".format(it)), dpi=300)
                plt.close()

            # Invert probababilistic fields for consistency with cripser sub-level set persistence
            combos = 1 - combos

            # Get barcodes using cripser in parallel without autograd            
            combos_arr = combos.detach().cpu().numpy().astype(np.float64)
            combo_shape = combos_arr.shape[1:]
            if parallel:
                raise ValueError("Ups this does not work here!")
                # with torch.no_grad():
                #     with Pool(len(prior)) as p:
                #         bcodes_arr = p.starmap(ph[construction], zip(combos_arr, max_dims))
            else:
                with torch.no_grad():
                    bcodes_arr = [ph[construction](combo, max_dim) for combo, max_dim in zip(combos_arr, max_dims)]

            #Save bcodes, we save here as we need to know where it starts and ends, bcodes only have the lifetime
            if (it==0 or it==num_its-1):
                with open(os.path.join(resPath, "bcodes_{}.pickle".format(it)), 'wb') as f:
                    pickle.dump(bcodes_arr, f)
                with open(os.path.join(resPath, "logits_{}.pickle".format(it)), 'wb') as f:
                    pickle.dump(combos_arr, f)
                with open(os.path.join(resPath, "roi_{}.pickle".format(it)), 'wb') as f:
                    pickle.dump(roi, f)

            # Get differentiable barcodes using autograd
            max_features = max([bcode_arr.shape[0] for bcode_arr in bcodes_arr])
            bcodes = torch.zeros([len(prior), max(max_dims), max_features], requires_grad=False, device=self.device)
            for c, (combo, bcode) in enumerate(zip(combos, bcodes_arr)):
                _, fin = self.get_differentiable_barcode(combo, bcode, combo_shape)
                for dim in range(len(spatial_xyz)):
                    bcodes[c, dim, :len(fin[dim])] = fin[dim]

            # Select features for the construction of the topological loss
            stacked_prior = torch.stack(list(prior.values()))
            stacked_prior.T[0] -= 1 # Since fundamental 0D component has infinite persistence
            matching = torch.zeros_like(bcodes, dtype=torch.uint8).detach()

            #Do not touch the ones I am not certain about the topo
            #matching is not bool anymore, now it is: 
            #0 = not matched/incorrect
            #1 = correct
            #2 = I do not know (so neither correct neither incorrect in loss)
            bcodes_arr = bcodes.detach().cpu().numpy()
            for c, combo in enumerate(stacked_prior):
                for dim in range(len(combo)):
                    if stacked_prior[c, dim] >= 0: # If user put a certain topology
                        matching[c, dim, slice(None, stacked_prior[c, dim])] = 1
                    else: # If user put -1 (dubious topo)
                        nTopos = np.count_nonzero(bcodes_arr[c,dim,:])
                        matching[c, dim, slice(None, nTopos)] = 2
            
            # Find total persistence of features which match (A) / violate (Z) the prior
            A = (1 - bcodes[matching==1]).sum()
            Z = bcodes[matching==0].sum()

            # Get similarity constraint
            mse = F.mse_loss(outputs, prediction)

            # Optimisation
            loss = A + Z + mse_lambda * mse
            loss.backward()
            optimiser.step()
    

    def predict_sliding_window_return_logits(self, input_image: torch.Tensor) -> Union[np.ndarray, torch.Tensor]:
        assert isinstance(input_image, torch.Tensor)
        self.network = self.network.to(self.device)
        # self.network.eval() # Now we pass here when we need to train so we enable eval outside when predicting

        empty_cache(self.device)

        # Autocast can be annoying
        # If the device_type is 'cpu' then it's slow as heck on some CPUs (no auto bfloat16 support detection)
        # and needs to be disabled.
        # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False
        # is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
        # So autocast will only be active if we have a cuda device.
        with torch.autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            assert input_image.ndim == 4, 'input_image must be a 4D np.ndarray or torch.Tensor (c, x, y, z)'

            if self.verbose: 
                print(f'Input shape: {input_image.shape}')
                print("step_size:", self.tile_step_size)
                print("mirror_axes:", self.allowed_mirroring_axes if self.use_mirroring else None)

            # if input_image is smaller than tile_size we need to pad it to tile_size.
            data, slicer_revert_padding = pad_nd_image(input_image, self.configuration_manager.patch_size,
                                                       'constant', {'value': 0}, True,
                                                       None)

            slicers = self._internal_get_sliding_window_slicers(data.shape[1:])

            if self.perform_everything_on_device and self.device != 'cpu':
                # we need to try except here because we can run OOM in which case we need to fall back to CPU as a results device
                try:
                    predicted_logits = self._internal_predict_sliding_window_return_logits(data, slicers,
                                                                                           self.perform_everything_on_device)
                except RuntimeError:
                    print(
                        'Prediction on device was unsuccessful, probably due to a lack of memory. Moving results arrays to CPU')
                    empty_cache(self.device)
                    predicted_logits = self._internal_predict_sliding_window_return_logits(data, slicers, False)
            else:
                predicted_logits = self._internal_predict_sliding_window_return_logits(data, slicers,
                                                                                       self.perform_everything_on_device)

            empty_cache(self.device)
            # revert padding
            predicted_logits = predicted_logits[(slice(None), *slicer_revert_padding[1:])]
        return predicted_logits


    def predict_logits_from_preprocessed_data(self, data: torch.Tensor, ofolder: str) -> torch.Tensor:
        """
        IMPORTANT! IF YOU ARE RUNNING THE CASCADE, THE SEGMENTATION FROM THE PREVIOUS STAGE MUST ALREADY BE STACKED ON
        TOP OF THE IMAGE AS ONE-HOT REPRESENTATION! SEE PreprocessAdapter ON HOW THIS SHOULD BE DONE!

        RETURNED LOGITS HAVE THE SHAPE OF THE INPUT. THEY MUST BE CONVERTED BACK TO THE ORIGINAL IMAGE SIZE.
        SEE convert_predicted_logits_to_segmentation_with_correct_shape
        """
        n_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        prediction = None

        for i, params in enumerate(self.list_of_parameters):

            # messing with state dict names...
            if not isinstance(self.network, OptimizedModule):
                if 'temperature' in params.keys():
                    self.network.temperature = params['temperature']
                    params_copy = copy.deepcopy(params)
                    del params_copy['temperature']
                    self.network.model.load_state_dict(params_copy)

                    # Here we need to train for PH without temperature
                    # So, this is ugly af but we define self.network as self.network.model and save net with temperature for after
                    prePH_temp_net = copy.deepcopy(self.network)
                    self.network = self.network.model

                    #We need to retrain the net for this fold to have persistent homology
                    phResPath = os.path.join(ofolder,"fold_{}".format(i))
                    if not os.path.exists(phResPath): pathlib.Path(phResPath).mkdir(parents=True, exist_ok=True)
                    self.multi_class_topological_post_processing(
                        inputs=data, 
                        resPath=phResPath,
                    )

                    #Once this is ready we  update self.network to have the new model topo trained and the original temp
                    # AWFUL but I needed to be fast
                    postPH_model_net = copy.deepcopy(self.network)
                    self.network = prePH_temp_net
                    self.network.model = postPH_model_net

                else:
                    self.network.load_state_dict(params)
            else:
                # self.network._orig_mod.load_state_dict(params)
                raise ValueError("You should not be here!")

            # Once we have retrained, we get back to inference
            self.network.eval()

            # why not leave prediction on device if perform_everything_on_device? Because this may cause the
            # second iteration to crash due to OOM. Grabbing that with try except cause way more bloated code than
            # this actually saves computation time
            with torch.no_grad():
                if prediction is None:
                    prediction = self.predict_sliding_window_return_logits(data).to('cpu')
                else:
                    prediction += self.predict_sliding_window_return_logits(data).to('cpu')

        if len(self.list_of_parameters) > 1:
            prediction /= len(self.list_of_parameters)

        if self.verbose: print('Prediction done')
        torch.set_num_threads(n_threads)
        return prediction
    

    def predict_single_npy_array(self, input_image: np.ndarray, image_properties: dict,
                                 segmentation_previous_stage: np.ndarray = None,
                                 output_file_truncated: str = None,
                                 save_or_return_probabilities: bool = False):
        """
        WARNING: SLOW. ONLY USE THIS IF YOU CANNOT GIVE NNUNET MULTIPLE IMAGES AT ONCE FOR SOME REASON.


        input_image: Make sure to load the image in the way nnU-Net expects! nnU-Net is trained on a certain axis
                     ordering which cannot be disturbed in inference,
                     otherwise you will get bad results. The easiest way to achieve that is to use the same I/O class
                     for loading images as was used during nnU-Net preprocessing! You can find that class in your
                     plans.json file under the key "image_reader_writer". If you decide to freestyle, know that the
                     default axis ordering for medical images is the one from SimpleITK. If you load with nibabel,
                     you need to transpose your axes AND your spacing from [x,y,z] to [z,y,x]!
        image_properties must only have a 'spacing' key!
        """
        if not os.path.exists(output_file_truncated): pathlib.Path(output_file_truncated).mkdir(parents=True, exist_ok=True)

        ppa = PreprocessAdapterFromNpy([input_image], [segmentation_previous_stage], [image_properties],
                                       [output_file_truncated],
                                       self.plans_manager, self.dataset_json, self.configuration_manager,
                                       num_threads_in_multithreaded=1, verbose=self.verbose)
        if self.verbose:
            print('preprocessing')
        dct = next(ppa)

        if self.verbose:
            print('predicting')
        predicted_logits = self.predict_logits_from_preprocessed_data(dct['data'], dct['ofile']).cpu()

        if self.verbose:
            print('resampling to original shape')
        if output_file_truncated is not None:
            export_prediction_from_logits(predicted_logits, dct['data_properties'], self.configuration_manager,
                                          self.plans_manager, self.dataset_json, output_file_truncated,
                                          save_or_return_probabilities)
        else:
            ret = convert_predicted_logits_to_segmentation_with_correct_shape(predicted_logits, self.plans_manager,
                                                                              self.configuration_manager,
                                                                              self.label_manager,
                                                                              dct['data_properties'],
                                                                              return_probabilities=
                                                                              save_or_return_probabilities)
            if save_or_return_probabilities:
                return ret[0], ret[1]
            else:
                return ret