mlcolvar.cvs.BaseCV
===================

.. currentmodule:: mlcolvar.cvs

.. autoclass:: BaseCV
   :members:                                   
   :show-inheritance:                           
   :inherited-members: Module,LightningModule                       

   
   .. automethod:: __init__

   
   .. rubric:: Methods

   .. autosummary::
   
      ~BaseCV.__init__
      ~BaseCV.configure_optimizers
      ~BaseCV.forward
      ~BaseCV.forward_cv
      ~BaseCV.initialize_blocks
      ~BaseCV.initialize_transforms
      ~BaseCV.on_fit_start
      ~BaseCV.parse_model
      ~BaseCV.parse_options
      ~BaseCV.setup
      ~BaseCV.test_step
      ~BaseCV.to_torchscript
      ~BaseCV.validation_step
   
   


..
   
   
   .. rubric:: Attributes

   .. autosummary::
   
      ~BaseCV.CHECKPOINT_HYPER_PARAMS_KEY
      ~BaseCV.CHECKPOINT_HYPER_PARAMS_NAME
      ~BaseCV.CHECKPOINT_HYPER_PARAMS_TYPE
      ~BaseCV.DEFAULT_BLOCKS
      ~BaseCV.MODEL_BLOCKS
      ~BaseCV.T_destination
      ~BaseCV.automatic_optimization
      ~BaseCV.call_super_init
      ~BaseCV.current_epoch
      ~BaseCV.device
      ~BaseCV.device_mesh
      ~BaseCV.dtype
      ~BaseCV.dump_patches
      ~BaseCV.example_input_array
      ~BaseCV.fabric
      ~BaseCV.global_rank
      ~BaseCV.global_step
      ~BaseCV.hparams
      ~BaseCV.hparams_initial
      ~BaseCV.local_rank
      ~BaseCV.logger
      ~BaseCV.loggers
      ~BaseCV.n_cvs
      ~BaseCV.on_gpu
      ~BaseCV.optimizer_name
      ~BaseCV.strict_loading
      ~BaseCV.trainer
      ~BaseCV.training
   
   

   