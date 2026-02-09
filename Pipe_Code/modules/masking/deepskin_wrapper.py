"""
DeepSkin Wrapper Module

This module provides a cached wrapper around the DeepSkin wound_segmentation function
to prevent TensorFlow retracing issues. The DeepSkin model is loaded once and reused
across multiple calls, significantly improving performance.

This wrapper is maintained in the main codebase rather than modifying the external
DeepSkin repository, ensuring compatibility across different machines.
"""

import numpy as np

# Global variable to cache the DeepSkin segmentation function and model
_cached_deepskin_func = None
_model_initialized = False


def get_cached_wound_segmentation():
    """
    Get or create a cached version of the DeepSkin wound_segmentation function.
    
    This function implements model caching to avoid TensorFlow retracing warnings.
    The model is loaded once on first call and reused for all subsequent calls.
    
    Returns
    -------
    function
        The wound_segmentation function with cached model
    """
    global _cached_deepskin_func, _model_initialized
    
    if _cached_deepskin_func is not None:
        return _cached_deepskin_func
    
    # Import deepskin (this will only happen once)
    from deepskin.segmentation import wound_segmentation as original_wound_segmentation
    from deepskin.model import deepskin_model, MODEL_CHECKPOINT
    from deepskin.checkpoints import download_model_weights
    import os
    
    # Create the model once
    model_cache = {'model': None}
    
    def cached_wound_segmentation(img: np.ndarray, tol: float = 0.5, verbose: bool = False) -> np.ndarray:
        """
        Cached version of wound_segmentation that reuses the same model instance.
        
        Parameters
        ----------
        img : np.ndarray
            Input image to analyze in RGB format
        tol : float (default := 0.5)
            Threshold to apply on the resulting mask for output binarization
        verbose : bool (default := False)
            Enable/Disable the logging of the steps
            
        Returns
        -------
        pred : np.ndarray
            Output image mask obtained by the model, in which the semantic meaning
            is organized as: background (channel 0), body (channel 1), wound (channel 2)
        """
        import cv2
        from time import time as now
        
        # Get or create the cached model
        if model_cache['model'] is None:
            if verbose:
                print('Loading DeepSkin model (first time only)...', end='', flush=True)
            
            # Build the model
            model = deepskin_model(verbose=False)
            
            # Load the weights
            import sys
            if hasattr(sys.modules['deepskin'], '__file__'):
                local = os.path.dirname(os.path.abspath(sys.modules['deepskin'].__file__))
            else:
                # Fallback for different import scenarios
                import deepskin
                local = os.path.dirname(os.path.abspath(deepskin.__file__))
            
            weightspath = os.path.join(local, '..', 'checkpoints', 'efficientnetb3_deepskin_semantic.h5')
            
            if not os.path.exists(weightspath):
                download_model_weights(
                    Id=MODEL_CHECKPOINT,
                    model_name='efficientnetb3_deepskin_semantic'
                )
            
            model.load_weights(weightspath)
            model_cache['model'] = model
            
            if verbose:
                print(' Done!', flush=True)
        
        model = model_cache['model']
        
        # Original segmentation logic
        tic = now()
        step = 'Perform the semantic image segmentation... '
        
        if verbose:
            print(f'{step}', end='\r', flush=True)
        
        # Get the model input shape
        _, h, w, c = model.input.shape
        
        # Pre-process the input
        if verbose:
            print(f'{step} [1/3] pre-process the input image', end='', flush=True)
        
        resized = cv2.resize(img, dsize=(h, w), interpolation=cv2.INTER_CUBIC)
        resized = np.float32(resized)
        resized *= 1. / 255
        resized = resized.reshape(1, *resized.shape)
        
        # Apply the segmentation model
        if verbose:
            print(f'\r{step} [2/3] apply the segmentation model', end='', flush=True)
        
        pred = model.predict(resized, verbose=0)
        pred = np.squeeze(pred)
        pred = np.where(pred > tol, 255, 0)
        pred = np.uint8(pred)
        
        # Post-process the mask
        if verbose:
            print(f'\r{step} [3/3] post-process the segmentation mask', end='', flush=True)
        
        pred = cv2.resize(pred, dsize=(img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST_EXACT)
        
        toc = now()
        if verbose:
            print(f'\r{step} [DONE] ({toc - tic:.3f} sec)' + ' ' * 30, flush=True)
        
        return pred
    
    _cached_deepskin_func = cached_wound_segmentation
    return _cached_deepskin_func


def wound_segmentation(img: np.ndarray, tol: float = 0.5, verbose: bool = False) -> np.ndarray:
    """
    Wrapper function for DeepSkin wound segmentation with model caching.
    
    This function prevents TensorFlow retracing by caching the model instance.
    Use this instead of directly importing from deepskin.segmentation.
    
    Parameters
    ----------
    img : np.ndarray
        Input image to analyze in RGB format
    tol : float (default := 0.5)
        Threshold to apply on the resulting mask for output binarization
    verbose : bool (default := False)
        Enable/Disable the logging of the steps
        
    Returns
    -------
    pred : np.ndarray
        Output image mask obtained by the model, in which the semantic meaning
        is organized as: background (channel 0), body (channel 1), wound (channel 2)
    """
    cached_func = get_cached_wound_segmentation()
    return cached_func(img=img, tol=tol, verbose=verbose)
