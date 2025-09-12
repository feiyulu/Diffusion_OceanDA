# --- config.py ---
# This file defines global configuration parameters for the diffusion model.
import torch
import json
import os

class Config:
    """
    A centralized configuration class for the ocean data assimilation project.
    """
    def __init__(
        self,
        test_id,
        data_shape=(25, 128, 128),
        use_salinity=False,

        # --- File Paths and Variable Names ---
        filepath_t=None,
        filepath_s=None,
        filepath_t_test=None,
        filepath_s_test=None,
        filepath_static=None, 
        filepath_mask=None,
        filepath_z_static=None,
        varname_t='T',
        varname_s='S',
        varname_lat='lat',
        varname_lon='lon',
        mask_varname='wet',
        filepath_t_clim=None,
        area_weight_varname='area_t',

        # --- Data Slicing and Subsetting ---
        # Specifies the spatial and temporal domain for experiments.
        depth_range=[0,25],
        lat_range=[26,154],
        lon_range=[120,248],
        training_years=[2013, 2014],
        training_day_interval=1,
    
        # --- Normalization Ranges ---
        # Defines the min/max values for normalizing physical data to the [0, 1] range.
        T_range=[-2,33],
        S_range=[32,37],

        # --- Core Diffusion Model Hyperparameters ---
        # Controls the noise schedule for the diffusion process.
        timesteps=1000,
        beta_start=1e-5,
        beta_end=0.01,

        # --- U-Net Architecture ---
        # Defines the structure of the neural network.
        architecture_style="factorized_2d",
        vertical_encoder_groups=[[0]], 
        vertical_latent_dims=[32], 
        base_unet_channels=64,
        channel_multipliers=(1, 2, 4, 8),
        attn_resolutions=(16,), 
        num_res_blocks=2,
        dropout_prob=0.1,

        # --- Training Parameters ---
        # Controls the optimization process.
        epochs=100,
        batch_size=8,
        learning_rate=1e-4,
        use_checkpointing=True,
        use_amp=True,
        use_lr_scheduler=True,
        lr_scheduler_T_max=100,
        lr_scheduler_eta_min=1e-6, 
        gradient_accumulation_steps=1,
        validation_split=0.1,
        save_model_after_training=True,
        save_interval=10,

        # --- Experiment Tracking (Weights & Biases) ---
        # Configuration for logging metrics and results with W&B.
        use_wandb=False,
        wandb_project="diffusion_ocean_da",
        wandb_entity="feiyulu-princeton",
        wandb_offline=False,

        # --- Conditioning Parameters ---
        conditioning_configs={
            "dayofyear": {"dim": 64},
            "co2": {"dim": 64}
        },
        co2_filepath=None,
        co2_varname='co2',
        co2_range=[320,450],
        location_embedding_types=["lon_cyclical", "cos_lat", "coriolis"],

        # --- NEW: Previous State Conditioning ---
        # A list of previous states to condition the model on.
        # Each entry specifies the time lag and which channels to use.
        # Example: [{"lag_days": 1, "channels": [0]}, {"lag_days": 5, "channels": [0]}]
        # This would use SST from 1 day ago and 5 days ago.
        previous_states=[],

        # --- Sampling Parameters ---
        # Controls how new samples are generated from the trained model.
        sampling_method='ddpm',
        ensemble_size=1,
        sampling_batch_size=4,
        sampling_steps=20,
        ddim_eta=0.0,
        use_full_ddpm_schedule=False,
        
        # --- Observation Settings ---
        # This list defines all possible observation sources (real or synthetic) that can be
        # used for conditional sampling (data assimilation).
        observation_sources=[],
            # {
            #     "name": "synthetic_argo",
            #     "type": "synthetic_profiles",
            #     "enabled": True,
            #     "num_profiles": 100,
            #     "guidance_strength": 0.8, # Overall strength of the correction
            #     "operator": "localized_innovation", # 'point_replacement', 'localized_innovation', or 'latent_blending'
            #     "localization_radius": 4 # Radius of influence in grid cells
            # },
            # {
            #     "name": "real_argo",
            #     "type": "real_argo",
            #     "enabled": False, # Disabled by default
            #     "filepath_template": "/scratch/cimes/feiyul/Ocean_Data/obs_data/argo/argo_{year}_interp25.nc",
            #     "time_window_days": 1,
            #     "guidance_strength": 0.8,
            #     "operator": "localized_innovation",
            #     "localization_radius": 4
            # },
            # {
            #     "name": "real_sst",
            #     "type": "real_sst",
            #     "enabled": False, # Disabled by default
            #     "filepath_template": "/scratch/cimes/feiyul/Ocean_Data/obs_data/sst/sst.day.{year}.regridded.nc",
            #     "target_channel": 0,
            #     "guidance_strength": 0.5,
            #     "operator": "localized_innovation",
            #     "localization_radius": 2
            # },
            # {
            #     "name": "synthetic_sst",
            #     "type": "synthetic_surface",
            #     "enabled": True, # Disabled by default
            #     "target_channel": 0, # e.g., Temperature
            #     "guidance_strength": 0.5,
            #     "operator": "localized_innovation",
            #     "subsample_fraction": 0.05, # Use only 5% of SST data for guidance
            #     "localization_radius": 2
            # }
            # ,
            # {
            #     "name": "resampling_argo",
            #     "type": "synthetic_profiles",
            #     "enabled": False, # Disabled by default
            #     "num_profiles": 50,
            #     "guidance_strength": 1.0,
            #     "operator": "resampling_guidance",
            #     "resampling_steps": 50, # Number of diffusion steps for resampling
            #     "resampling_method": "dpm-solver++", # 'ddpm' or 'dpm-solver++'
            #     "resampling_iterations": 1 # Number of refinement iterations
            # }

        # --- Evaluation Settings ---
        # Defines which days to run sampling on and which depth levels to plot.
        sample_years=[2024,2025],
        sample_days=[[0]],
        generate_training_animation=False,
        plot_depth_levels=[0]
        ):
        
        self.test_id = test_id
        # ... (rest of the __init__ is the same, no changes needed)
        self.data_shape = data_shape
        self.use_salinity = use_salinity
        self.channels = 2 if self.use_salinity else 1
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.filepath_t = [filepath_t.format(year=year) for year in range(training_years[0],training_years[1]+1)]
        self.filepath_s = [filepath_s.format(year=year) for year in range(training_years[0],training_years[1]+1)] if self.use_salinity and filepath_s else None
        self.filepath_t_test = [filepath_t_test.format(year=year) for year in range(sample_years[0],sample_years[1]+1)]
        self.filepath_s_test = [filepath_s_test.format(year=year) for year in range(sample_years[0],sample_years[1]+1)] if self.use_salinity and filepath_s_test else None
        
        self.filepath_static = filepath_static
        self.filepath_mask = filepath_mask
        self.filepath_z_static = filepath_z_static
        self.mask_varname = mask_varname
        self.varname_t = varname_t
        self.varname_s = varname_s
        self.varname_lat = varname_lat
        self.varname_lon = varname_lon
        self.filepath_t_clim = filepath_t_clim
        self.area_weight_varname = area_weight_varname
        
        self.depth_range = depth_range
        self.lat_range = lat_range
        self.lon_range = lon_range
        self.training_day_range=[f"{training_years[0]}0101",f"{training_years[1]}1231"]
        self.training_day_interval = training_day_interval
        self.sample_days = sample_days
        self.T_range = T_range
        self.S_range = S_range

        self.scratch_dir = "/scratch/cimes/feiyul/Diffusion_OceanDA"
        self.output_dir = f"{self.scratch_dir}/{self.test_id}"
        os.makedirs(self.output_dir, exist_ok=True)

        self.timesteps = timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        
        self.architecture_style = architecture_style
        self.vertical_encoder_groups = vertical_encoder_groups
        self.vertical_latent_dims = vertical_latent_dims
        
        self.base_unet_channels = base_unet_channels
        self.channel_multipliers = channel_multipliers
        self.attn_resolutions = attn_resolutions
        self.num_res_blocks = num_res_blocks
        self.dropout_prob = dropout_prob

        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.use_checkpointing = use_checkpointing
        self.use_amp = use_amp
        self.use_lr_scheduler = use_lr_scheduler
        self.lr_scheduler_T_max = lr_scheduler_T_max or epochs
        self.lr_scheduler_eta_min = lr_scheduler_eta_min
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.validation_split = validation_split
        self.save_model_after_training = save_model_after_training
        self.save_interval = save_interval

        self.use_wandb = use_wandb
        self.wandb_project = wandb_project
        self.wandb_entity = wandb_entity
        self.wandb_offline = wandb_offline

        self.conditioning_configs = conditioning_configs
        self.co2_filepath = co2_filepath
        self.co2_varname = co2_varname
        self.co2_range = co2_range
        self.location_embedding_types = location_embedding_types
        self.previous_states = previous_states
        
        self.location_embedding_channels = 0
        if self.location_embedding_types:
            type_counts = {"lat": 1, "lon": 1, "lon_cyclical": 2, "cos_lat": 1, "coriolis": 1, "depth_ocean": 1, "grid_area": 1}
            for emb_type in self.location_embedding_types:
                self.location_embedding_channels += type_counts.get(emb_type, 0)
 
        self.sampling_method = sampling_method
        self.ensemble_size = ensemble_size
        self.sampling_batch_size = sampling_batch_size
        self.sampling_steps = sampling_steps
        self.ddim_eta = ddim_eta
        self.use_full_ddpm_schedule = use_full_ddpm_schedule
        
        self.observation_sources = observation_sources

        self.sample_years = sample_years
        self.sample_days = sample_days
        self.generate_training_animation = generate_training_animation
        self.plot_depth_levels = plot_depth_levels

        self.model_checkpoint_dir = f"{self.output_dir}/checkpoints"
        os.makedirs(self.model_checkpoint_dir, exist_ok=True)
        self.loss_plot_dir = f"{self.output_dir}/loss_plots"
        os.makedirs(self.loss_plot_dir, exist_ok=True)
        self.sample_plot_dir = f"{self.output_dir}/sample_plots"
        os.makedirs(self.sample_plot_dir, exist_ok=True)
        self.training_animation_path = f"{self.output_dir}/training_data_animation_{self.test_id}.gif"

    @classmethod
    def from_json_file(cls, filepath):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Config file not found: {filepath}")
        with open(filepath, 'r') as f:
            settings = json.load(f)
        for key in ['data_shape', 'channel_multipliers', 'attn_resolutions', 'pixel_shuffle_size', 'vertical_encoder_groups', 'vertical_latent_dims', 'observation_sources']:
            if key in settings and isinstance(settings[key], list):
                pass
        return cls(**settings)

    def to_json_file(self, filepath):
        settings = self.__dict__.copy()
        non_serializable_keys = ['device', 'model_checkpoint_dir', 'loss_plot_dir', 'sample_plot_dir', 'training_animation_path']
        for key in non_serializable_keys:
            if key in settings:
                del settings[key]
        for key, value in settings.items():
            if isinstance(value, tuple):
                settings[key] = list(value)
        with open(filepath, 'w') as f:
            json.dump(settings, f, indent=4)
        print(f"Configuration saved to {filepath}")
