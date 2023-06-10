
''' Code from 
https://github.com/nick-byrne/topological-losses and https://github.com/MIC-DKFZ/nnUNet 

based on 

N. Byrne, J. R. Clough, I. Valverde, G. Montana and A. P. King, 
"A persistent homology-based topological loss for CNN-based multi-class segmentation of CMR," 
in IEEE Transactions on Medical Imaging, 2022, doi: 10.1109/TMI.2022.3203309. and 

and 

Isensee, F., Jaeger, P. F., Kohl, S. A., Petersen, J., & Maier-Hein, K. H. (2021).
"nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation."
Nature methods, 18(2), 203-211.

'''

import inspect
import os
import shutil
from copy import deepcopy
from typing import Tuple, Union, List
from multiprocessing import Pool
import warnings
import matplotlib.pyplot as plt
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from batchgenerators.transforms.utility_transforms import NumpyToTensor
from batchgenerators.utilities.file_and_folder_operations import join, isfile, maybe_mkdir_p, save_json
from nnunetv2.inference.predict_from_raw_data import PreprocessAdapter, auto_detect_available_folds, load_what_we_need
from nnunetv2.inference.export_prediction import export_prediction_from_softmax
from nnunetv2.inference.sliding_window_prediction import compute_gaussian
from nnunetv2.utilities.json_export import recursive_fix_for_json_export
from nnunetv2.utilities.utils import create_lists_from_splitted_dataset_folder
from nnunetv2.utilities.helpers import dummy_context
from acvl_utils.cropping_and_padding.padding import pad_nd_image
from nnunetv2.inference.sliding_window_prediction import get_sliding_window_generator, maybe_mirror_and_predict

import cripser as crip
import tcripser as trip


def crip_wrapper(X, D):
    return crip.computePH(X, maxdim=D)

def trip_wrapper(X, D):
    return trip.computePH(X, maxdim=D)

def get_roi(X, thresh=0.01):
    true_points = torch.nonzero(X >= thresh)
    corner1 = true_points.min(dim=0)[0]
    corner2 = true_points.max(dim=0)[0]
    roi = [slice(None, None)] + [slice(c1, c2 + 1) for c1, c2 in zip(corner1, corner2)]
    return roi

def getBarcodes(tensor, prior, max_dims, ph, construction, parallel=True):
    # Build class/combination-wise (c-wise) image tensor for prior
    tmp = []
    for c in prior.keys():
        if c.dim()== 1:
            tmp.append(tensor[c].sum(0))
        else:
            raise ValueError("Wrong dim")
    combos = torch.stack(tmp)

    # Invert probababilistic fields for consistency with cubical ripser sub-level set persistence
    combos = 1 - combos #TODO check here the max may not be 1 in the vol and then we might have to change this

    # Get barcodes using cripser in parallel without autograd            
    combos_arr = combos.detach().cpu().numpy().astype(np.float64)
    if parallel:
        with torch.no_grad():
            with Pool(len(prior)) as p:
                bcodes_arr = p.starmap(ph[construction], zip(combos_arr, max_dims))
    else:
        with torch.no_grad():
            bcodes_arr = [ph[construction](combo, max_dim) for combo, max_dim in zip(combos_arr, max_dims)]
            
    return bcodes_arr  


def checkIndexing(indexes, shape):
    shapeUpLimit = np.array(shape)[np.newaxis,:] - 1
    shapeDownLimit = np.array((0,0,0))[np.newaxis,:]
    idxsOut    = (indexes > shapeUpLimit).nonzero()
    indexes[idxsOut] = shapeUpLimit[0,idxsOut[1]]
    idxsOut    = (indexes < shapeDownLimit).nonzero()
    indexes[idxsOut] = shapeDownLimit[0,idxsOut[1]]


def get_differentiable_barcode(tensor, barcode, shape):
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
    checkIndexing(indexes, shape)
    inf_birth = tensor[tuple(indexes.T)]
    
    # Calculate lifetimes of finite features
    indexes = fin[:, 3:3+tensor.ndim].astype(np.int64)
    checkIndexing(indexes, shape)
    births = tensor[tuple(indexes.T)]
    indexes = fin[:, 6:6+tensor.ndim].astype(np.int64)
    checkIndexing(indexes, shape)
    deaths = tensor[tuple(indexes.T)]
    delta_p = (deaths - births)
    
    # Split finite features by dimension
    delta_p = [delta_p[fin[:, 0] == d] for d in range(tensor.ndim)]
    
    # Sort finite features by persistence
    delta_p = [torch.sort(d, descending=True)[0] for d in delta_p]
    
    return inf_birth, delta_p

def multi_class_topological_post_processing(
    inputs, model, priors,
    num_seg_heads, patch_size,
    inference_allowed_mirroring_axes,
    tile_step_size,
    use_gaussian,
    precomputed_gaussian,
    perform_everything_on_gpu,
    device,
    saveCombosPath,
    sample,
    lr=1e-5, mse_lambda=1000,
    opt=torch.optim.Adam, num_its=100, construction='N', thresh=None, parallel=True,
    ):
    '''Performs topological post-processing.
    
    Arguments:
        REQUIRED
        inputs       - PyTorch tensor - [1, number of classes] + [spatial dimensions (2D or 3D)]
        model        - Pre-trained CNN as PyTorch module (without final activation)
        prior        - Topological prior as dictionary:
                       keys are tuples specifying the channel(s) of inputs
                       values are tuples specifying the desired Betti numbers
        lr           - Learning rate for SGD optimiser
        mse_lambda   - Weighting for similarity constraint
        
        OPTIONAL [default]
        opt          - PyTorch optimiser [torch.optim.Adam]
        num_its      - Iterable of number iterations(s) to run for each scale [100]
        construction - Either '0' (4 (2D) or 6 (3D) connectivity) or 'N' (8 (2D) or 26 (3D) connectivity) ['0']
        thresh       - Threshold at which to define the foreground ROI for topological post-processing
        We Added the extra params for predict_sliding_window_return_logits_ph(...)
    '''
    # Get image properties
    spatial_xyz = list(inputs.shape[1:])
    
    # Get raw prediction
    model.eval()
    with torch.no_grad():
        pred_unet = predict_sliding_window_return_logits_ph(
                    model, inputs, 
                    num_seg_heads,
                    patch_size,
                    mirror_axes=inference_allowed_mirroring_axes,
                    tile_step_size=tile_step_size,
                    use_gaussian=use_gaussian,
                    precomputed_gaussian=precomputed_gaussian,
                    perform_everything_on_gpu=perform_everything_on_gpu,
                    verbose=False,
                    device=device)
        
        pred_unet = torch.softmax(pred_unet, 0)
        # pred_unet = torch.softmax(model(inputs), 1).detach().squeeze()   pred_unet [C, D,H,W]

    #Determine the prior mased on class mvo
    isThereMvo = pred_unet.argmax(dim=0)
    if isThereMvo.max()==4:
        prior = priors[1]
    elif isThereMvo.max()==3:
        prior = priors[0]
    else:
        raise ValueError('Class 3 MI is not present in prediction!!,' 
        'Maybe add priors in case only a variable number of classes are present for test image')

    # If appropriate, choose ROI for topological consideration
    if thresh:
        roi = get_roi(pred_unet[1:].sum(0).squeeze(), thresh)
    else:
        roi = [slice(None, None)] + [slice(None, None) for dim in range(len(spatial_xyz))]
    
    # Initialise topological model and optimiser
    model_topo = deepcopy(model)
    model_topo.train()
    optimiser = opt(model_topo.parameters(), lr=lr)
    
    # Inspect prior and convert to tensor
    max_dims = [len(b) for b in prior.values()]
    prior = {torch.tensor(c): torch.tensor(b) for c, b in prior.items()}
    
    # Set mode of cubical complex construction
    ph = {'0': crip_wrapper, 'N': trip_wrapper}

    for it in range(num_its):

        # Reset gradients
        optimiser.zero_grad()

        # Get current prediction
        # outputs = torch.softmax(model_topo(inputs), 1).squeeze()
        outputs = predict_sliding_window_return_logits_ph(
                    model_topo, inputs, 
                    num_seg_heads,
                    patch_size,
                    mirror_axes=inference_allowed_mirroring_axes,
                    tile_step_size=tile_step_size,
                    use_gaussian=use_gaussian,
                    precomputed_gaussian=precomputed_gaussian,
                    perform_everything_on_gpu=perform_everything_on_gpu,
                    verbose=False,
                    device=device)
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

        
        if it==0:
            if saveCombosPath:
                combosArr = combos.cpu().detach().numpy()
                nImgs = combosArr.shape[0]+1
                ncols=3
                nrows = int(np.ceil(nImgs/ncols))
                f, a = plt.subplots(nrows, ncols)
                j=0
                i=0
                for nImg in range(nImgs-1):
                    a[j,i].imshow(combosArr[nImg,10,:,:])
                    a[j,i].axis('off')
                    i+=1
                    if i % ncols == 0: 
                        j += 1
                        i = 0
                a[j,i].imshow(inputs[0,10,:,:])
                plt.savefig(os.path.join(saveCombosPath, sample))

        # Invert probababilistic fields for consistency with cripser sub-level set persistence
        combos = 1 - combos

        # Get barcodes using cripser in parallel without autograd            
        combos_arr = combos.detach().cpu().numpy().astype(np.float64)
        combo_shape = combos_arr.shape[1:]
        if parallel:
            with torch.no_grad():
                with Pool(len(prior)) as p:
                    bcodes_arr = p.starmap(ph[construction], zip(combos_arr, max_dims))
        else:
            with torch.no_grad():
                bcodes_arr = [ph[construction](combo, max_dim) for combo, max_dim in zip(combos_arr, max_dims)]

        # Get differentiable barcodes using autograd
        max_features = max([bcode_arr.shape[0] for bcode_arr in bcodes_arr])
        bcodes = torch.zeros([len(prior), max(max_dims), max_features], requires_grad=False, device=device)
        for c, (combo, bcode) in enumerate(zip(combos, bcodes_arr)):
            _, fin = get_differentiable_barcode(combo, bcode, combo_shape)
            for dim in range(len(spatial_xyz)):
                bcodes[c, dim, :len(fin[dim])] = fin[dim]

        # Select features for the construction of the topological loss
        stacked_prior = torch.stack(list(prior.values()))
        stacked_prior.T[0] -= 1 # Since fundamental 0D component has infinite persistence
        matching = torch.zeros_like(bcodes).detach().bool()
        for c, combo in enumerate(stacked_prior):
            for dim in range(len(combo)):
                matching[c, dim, slice(None, stacked_prior[c, dim])] = True

        # Find total persistence of features which match (A) / violate (Z) the prior
        A = (1 - bcodes[matching]).sum()
        Z = bcodes[~matching].sum()

        # Get similarity constraint
        mse = F.mse_loss(outputs, pred_unet)

        # Optimisation
        loss = A + Z + mse_lambda * mse
        loss.backward()
        optimiser.step()

    return model_topo


def predict_sliding_window_return_logits_ph(network: nn.Module,
                                         input_image: Union[np.ndarray, torch.Tensor],
                                         num_segmentation_heads: int,
                                         tile_size: Tuple[int, ...],
                                         mirror_axes: Tuple[int, ...] = None,
                                         tile_step_size: float = 0.5,
                                         use_gaussian: bool = True,
                                         precomputed_gaussian: torch.Tensor = None,
                                         perform_everything_on_gpu: bool = True,
                                         verbose: bool = True,
                                         device: torch.device = torch.device('cuda')) -> Union[np.ndarray, torch.Tensor]:
    if perform_everything_on_gpu:
        assert device.type == 'cuda', 'Can use perform_everything_on_gpu=True only when device="cuda"'
    network = network.to(device)

    # Autocast is a little bitch.
    # If the device_type is 'cpu' then it's slow as heck and needs to be disabled.
    # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
    # So autocast will only be active if we have a cuda device.
    with torch.autocast(device.type, enabled=True) if device.type == 'cuda' else dummy_context():
        assert len(input_image.shape) == 4, 'input_image must be a 4D np.ndarray or torch.Tensor (c, x, y, z)'

        if not torch.cuda.is_available():
            if perform_everything_on_gpu:
                print('WARNING! "perform_everything_on_gpu" was True but cuda is not available! Set it to False...')
            perform_everything_on_gpu = False

        results_device = device if perform_everything_on_gpu else torch.device('cpu')

        if verbose: print("step_size:", tile_step_size)
        if verbose: print("mirror_axes:", mirror_axes)

        if not isinstance(input_image, torch.Tensor):
            # pytorch will warn about the numpy array not being writable. This doesnt matter though because we
            # just want to read it. Suppress the warning in order to not confuse users...
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                input_image = torch.from_numpy(input_image)

        # if input_image is smaller than tile_size we need to pad it to tile_size.
        data, slicer_revert_padding = pad_nd_image(input_image, tile_size, 'constant', {'value': 0}, True, None)

        if use_gaussian:
            gaussian = torch.from_numpy(
                compute_gaussian(tile_size, sigma_scale=1. / 8)) if precomputed_gaussian is None else precomputed_gaussian
            gaussian = gaussian.half()
            # make sure nothing is rounded to zero or we get division by zero :-(
            mn = gaussian.min()
            if mn == 0:
                gaussian.clip_(min=mn)

        slicers = get_sliding_window_generator(data.shape[1:], tile_size, tile_step_size, verbose=verbose)

        # preallocate results and num_predictions. Move everything to the correct device
        predicted_logits = torch.zeros((num_segmentation_heads, *data.shape[1:]), dtype=torch.half, device=results_device)
        n_predictions = torch.zeros(data.shape[1:], dtype=torch.half, device=results_device)
        gaussian = gaussian.to(results_device)

        for sl in slicers:
            workon = data[sl][None]
            workon = workon.to(device, non_blocking=False)

            prediction = maybe_mirror_and_predict(network, workon, mirror_axes)[0].to(results_device)

            predicted_logits[sl] += (prediction * gaussian if use_gaussian else prediction)
            n_predictions[sl[1:]] += (gaussian if use_gaussian else 1)

        predicted_logits /= n_predictions

    return predicted_logits[tuple([slice(None), *slicer_revert_padding[1:]])]



def predict_from_raw_data_ph(list_of_lists_or_source_folder: Union[str, List[List[str]]],
                        output_folder: str,
                        model_training_output_dir: str,

                        phPriors: List[dict],
                        phConstruction: str = 'N',
                        phThres: float = 0.5,
                        phParallel: bool = False,
                        saveCombosPath: str = None,

                        use_folds: Union[Tuple[int, ...], str] = None,
                        tile_step_size: float = 0.5,     
                        use_gaussian: bool = True,
                        use_mirroring: bool = True,
                        perform_everything_on_gpu: bool = True,
                        verbose: bool = True,
                        save_probabilities: bool = False,
                        overwrite: bool = True,
                        checkpoint_name: str = 'checkpoint_final.pth',
                        num_processes_preprocessing: int = 3,
                        folder_with_segs_from_prev_stage: str = None,
                        num_parts: int = 1,
                        part_id: int = 0,
                        device: torch.device = torch.device('cuda')):

    if device.type == 'cuda':
        device = torch.device(type='cuda', index=0)  # set the desired GPU with CUDA_VISIBLE_DEVICES!

    if device.type != 'cuda':
        perform_everything_on_gpu = False

    # let's store the input arguments so that its clear what was used to generate the prediction
    my_init_kwargs = {}
    for k in inspect.signature(predict_from_raw_data_ph).parameters.keys():
        if k != "phPriors": #dictionaries cannot be saved !
            my_init_kwargs[k] = locals()[k]
    my_init_kwargs = deepcopy(my_init_kwargs)  # let's not unintentionally change anything in-place. Take this as a
    # safety precaution.
    recursive_fix_for_json_export(my_init_kwargs)
    maybe_mkdir_p(output_folder)
    save_json(my_init_kwargs, join(output_folder, 'predict_from_raw_data_args.json'))

    if use_folds is None:
        use_folds = auto_detect_available_folds(model_training_output_dir, checkpoint_name)

    # load all the stuff we need from the model_training_output_dir
    parameters, configuration_manager, inference_allowed_mirroring_axes, \
    plans_manager, dataset_json, network, trainer_name = \
        load_what_we_need(model_training_output_dir, use_folds, checkpoint_name)

    # check if we need a prediction from the previous stage
    if configuration_manager.previous_stage_name is not None:
        if folder_with_segs_from_prev_stage is None:
            raise NotImplementedError(f'WARNING: The requested configuration is a cascaded model and requires predctions from the '
                  f'previous stage! folder_with_segs_from_prev_stage was not provided. Trying to run the '
                  f'inference of the previous stage...')


    # sort out input and output filenames
    if isinstance(list_of_lists_or_source_folder, str):
        list_of_lists_or_source_folder = create_lists_from_splitted_dataset_folder(list_of_lists_or_source_folder,
                                                                                   dataset_json['file_ending'])
    print(f'There are {len(list_of_lists_or_source_folder)} cases in the source folder')
    list_of_lists_or_source_folder = list_of_lists_or_source_folder[part_id::num_parts]
    caseids = [os.path.basename(i[0])[:-(len(dataset_json['file_ending']) + 5)] for i in list_of_lists_or_source_folder]
    print(f'I am process {part_id} out of {num_parts} (max process ID is {num_parts - 1}, we start counting with 0!)')
    print(f'There are {len(caseids)} cases that I would like to predict')

    output_filename_truncated = [join(output_folder, i) for i in caseids]
    seg_from_prev_stage_files = [join(folder_with_segs_from_prev_stage, i + dataset_json['file_ending']) if
                                 folder_with_segs_from_prev_stage is not None else None for i in caseids]
    # remove already predicted files form the lists
    if not overwrite:
        tmp = [isfile(i + dataset_json['file_ending']) for i in output_filename_truncated]
        not_existing_indices = [i for i, j in enumerate(tmp) if not j]

        output_filename_truncated = [output_filename_truncated[i] for i in not_existing_indices]
        list_of_lists_or_source_folder = [list_of_lists_or_source_folder[i] for i in not_existing_indices]
        seg_from_prev_stage_files = [seg_from_prev_stage_files[i] for i in not_existing_indices]
        print(f'overwrite was set to {overwrite}, so I am only working on cases that haven\'t been predicted yet. '
              f'That\'s {len(not_existing_indices)} cases.')
        # caseids = [caseids[i] for i in not_existing_indices]

    # placing this into a separate function doesnt make sense because it needs so many input variables...
    preprocessor = configuration_manager.preprocessor_class(verbose=verbose)
    # hijack batchgenerators, yo
    # we use the multiprocessing of the batchgenerators dataloader to handle all the background worker stuff. This
    # way we don't have to reinvent the wheel here.
    num_processes = max(1, min(num_processes_preprocessing, len(list_of_lists_or_source_folder)))
    ppa = PreprocessAdapter(list_of_lists_or_source_folder, seg_from_prev_stage_files, preprocessor,
                            output_filename_truncated, plans_manager, dataset_json,
                            configuration_manager, num_processes)
    mta = MultiThreadedAugmenter(ppa, NumpyToTensor(), num_processes, 1, None, pin_memory=device.type == 'cuda')
    # mta = SingleThreadedAugmenter(ppa, NumpyToTensor())

    # precompute gaussian
    inference_gaussian = torch.from_numpy(
        compute_gaussian(configuration_manager.patch_size)).half()
    if perform_everything_on_gpu:
        inference_gaussian = inference_gaussian.to(device)

    # num seg heads is needed because we need to preallocate the results in predict_sliding_window_return_logits
    label_manager = plans_manager.get_label_manager(dataset_json)
    num_seg_heads = label_manager.num_segmentation_heads

    # go go go
    # No multiprocessing and with grads as we need to retrain example-wise
    network = network.to(device)

    for preprocessed in mta:
        data = preprocessed['data']
        if isinstance(data, str):
            delfile = data
            data = torch.from_numpy(np.load(data))
            os.remove(delfile)

        ofile = preprocessed['ofile']
        print(f'\nPredicting {os.path.basename(ofile)}:')
        print(f'perform_everything_on_gpu: {perform_everything_on_gpu}')

        properties = preprocessed['data_properites']

        # we have some code duplication here but this allows us to run with perform_everything_on_gpu=True as
        # default and not have the entire program crash in case of GPU out of memory. Neat. That should make
        # things a lot faster for some datasets.
        prediction = None
        start_image_pred_timer = time.time()
        for net_fold_num, params in enumerate(parameters):
            network.load_state_dict(params)
            
            #We need to retrain the net for this fold to have persistent homology
            network_TP = multi_class_topological_post_processing(
                inputs=data, 
                model=network, 
                priors=phPriors,
                num_seg_heads=num_seg_heads,
                patch_size=configuration_manager.patch_size,
                inference_allowed_mirroring_axes= inference_allowed_mirroring_axes if use_mirroring else None,
                tile_step_size=tile_step_size,
                use_gaussian=use_gaussian,
                precomputed_gaussian=inference_gaussian,
                perform_everything_on_gpu=perform_everything_on_gpu,
                device=device,
                saveCombosPath=saveCombosPath,
                sample="{}_fold{}".format(os.path.basename(ofile), net_fold_num),
                lr=1e-5, mse_lambda=1000, opt=torch.optim.Adam, 
                num_its=100, construction=phConstruction, 
                thresh=phThres, parallel=phParallel
            )

            if prediction is None:
                prediction = predict_sliding_window_return_logits_ph(
                    network_TP, data, num_seg_heads,
                    configuration_manager.patch_size,
                    mirror_axes=inference_allowed_mirroring_axes if use_mirroring else None,
                    tile_step_size=tile_step_size,
                    use_gaussian=use_gaussian,
                    precomputed_gaussian=inference_gaussian,
                    perform_everything_on_gpu=perform_everything_on_gpu,
                    verbose=verbose,
                    device=device)
            else:
                prediction += predict_sliding_window_return_logits_ph(
                    network_TP, data, num_seg_heads,
                    configuration_manager.patch_size,
                    mirror_axes=inference_allowed_mirroring_axes if use_mirroring else None,
                    tile_step_size=tile_step_size,
                    use_gaussian=use_gaussian,
                    precomputed_gaussian=inference_gaussian,
                    perform_everything_on_gpu=perform_everything_on_gpu,
                    verbose=verbose,
                    device=device)
        if len(parameters) > 1:
            prediction /= len(parameters)

        print('Prediction done, transferring to CPU if needed')
        prediction = prediction.detach().to('cpu').numpy()
        export_prediction_from_softmax(prediction, properties, configuration_manager, plans_manager, dataset_json, ofile, save_probabilities)
        print(f'done with {os.path.basename(ofile)} in {time.time()-start_image_pred_timer} s')

    # we need these two if we want to do things with the predictions like for example apply postprocessing
    shutil.copy(join(model_training_output_dir, 'dataset.json'), join(output_folder, 'dataset.json'))
    shutil.copy(join(model_training_output_dir, 'plans.json'), join(output_folder, 'plans.json'))