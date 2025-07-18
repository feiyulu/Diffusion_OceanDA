# --- config.py ---
# This file defines global configuration parameters for the diffusion model.
import torch
import json
import os

class Config:
    def __init__(
        self,
        test_id,
        image_size=(128, 128),
        use_salinity=False,

        # --- File Paths and Variable Names ---
        filepath_t=None,
        filepath_s=None,
        filepath_t_test=None,
        filepath_s_test=None,
        filepath_static=None,
        varname_t='sst',
        varname_s='sss',

        # --- Data Slicing and Subsetting ---
        lat_range=[26,154],
        lon_range=[120,248],
        training_years=[2013, 2014],
        training_day_interval=1,
    
        # --- Normalization Ranges ---
        T_range=[-2,33],
        S_range=[32,37],

        # --- Core Diffusion Model Hyperparameters ---
        timesteps=1000,
        beta_start=1e-5,
        beta_end=0.01,

        # --- U-Net Architecture ---
        base_unet_channels=32,

        # --- Training Parameters ---
        epochs=100,
        batch_size=200,
        learning_rate=1e-4,
        use_lr_scheduler=True,
        lr_scheduler_T_max=100,
        lr_scheduler_eta_min=1e-6, 
        gradient_accumulation_steps=1,
        channel_wise_normalization=True,
        validation_split=0.1,
        save_model_after_training=True,
        save_interval=10,

        # --- Experiment Tracking (Weights & Biases) ---
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

        location_embedding_types=[
            "lat", "lon_cyclical", "cos_lat", "coriolis", "ocean_depth"
            ],

        # --- Sampling Parameters ---
        load_model_for_sampling=False,
        sampling_method='ddpm',
        ensemble_size=1,
        sampling_steps=20,
        ddim_eta=0.0,
        observation_fidelity_weight=1.0,

        use_real_observations=False,
        observation_path_template="/path/to/obs_data/argo/argo_{year}_interp.nc",
        observation_time_window_days=3,
        observation_operator="nearest_neighbor",

        observation_samples=[1000],
        sample_years=[2024,2025],
        sample_days=[[0]],
        ):
        
        # --- Basic Setup ---
        self.test_id = test_id
        self.image_size = image_size
        self.use_salinity = use_salinity
        self.channels = 2 if self.use_salinity else 1
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # --- Path and Data Settings ---
        self.filepath_t = [
            filepath_t.format(year=year) for year in range(training_years[0],training_years[1]+1)
        ]
        self.filepath_s = [
            filepath_s.format(year=year) for year in range(training_years[0],training_years[1]+1)
        ] if self.use_salinity else None
        self.filepath_t_test = [
            filepath_t_test.format(year=year) for year in range(sample_years[0],sample_years[1]+1)
        ]
        self.filepath_s_test = [
            filepath_s_test.format(year=year) for year in range(sample_years[0],sample_years[1]+1)
        ] if self.use_salinity else None
        self.filepath_static = filepath_static
        self.varname_t = varname_t
        self.varname_s = varname_s if self.use_salinity else None
        self.lat_range = lat_range
        self.lon_range = lon_range
        self.training_day_range=[f"{training_years[0]}0101",f"{training_years[1]}1231"]
        self.training_day_interval = training_day_interval
        self.sample_days = sample_days
        self.T_range = T_range
        self.S_range = S_range if self.use_salinity else None

        # --- Output Directory Management ---
        self.output_dir = f"/scratch/cimes/feiyul/Diffusion_OceanDA/{self.test_id}"
        if not os.path.isdir(self.output_dir):
            os.mkdir(self.output_dir)

        # --- Store Model and Training Settings ---
        self.timesteps = timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.base_unet_channels = base_unet_channels
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.use_lr_scheduler = use_lr_scheduler
        self.lr_scheduler_T_max = lr_scheduler_T_max if lr_scheduler_T_max is not None else epochs
        self.lr_scheduler_eta_min = lr_scheduler_eta_min
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.channel_wise_normalization = channel_wise_normalization
        self.validation_split = validation_split
        self.save_model_after_training = save_model_after_training
        self.save_interval = save_interval

        # --- Store Experiment Tracking Settings ---
        self.use_wandb = use_wandb
        self.wandb_project = wandb_project
        self.wandb_entity = wandb_entity
        self.wandb_offline=wandb_offline,

        # --- Store Conditioning Settings ---
        self.conditioning_configs = conditioning_configs
        self.co2_filepath = co2_filepath
        self.co2_varname = co2_varname
        self.co2_range = co2_range

        self.location_embedding_types = location_embedding_types
        
        # Dynamically calculate the number of location embedding channels based on the list.
        self.location_embedding_channels = 0
        if self.location_embedding_types:
            if "lat" in self.location_embedding_types: self.location_embedding_channels += 1
            if "lon" in self.location_embedding_types: self.location_embedding_channels += 1
            if "lon_cyclical" in self.location_embedding_types: self.location_embedding_channels += 2
            if "cos_lat" in self.location_embedding_types: self.location_embedding_channels += 1
            if "coriolis" in self.location_embedding_types: self.location_embedding_channels += 1
            if "ocean_depth" in self.location_embedding_types: self.location_embedding_channels += 1
            if "grid_area" in self.location_embedding_types: self.location_embedding_channels += 1
 
        # --- Store Sampling Settings ---
        self.load_model_for_sampling = load_model_for_sampling
        self.sampling_method = sampling_method
        self.ensemble_size = ensemble_size
        self.sampling_steps = sampling_steps
        self.ddim_eta = ddim_eta
        self.observation_fidelity_weight = observation_fidelity_weight
        self.observation_samples = observation_samples

        # --- Dynamically Generated Paths ---
        # These paths are constructed based on the test_id and other settings.
        self.model_checkpoint_dir = f"{self.output_dir}/checkpoints"
        os.makedirs(self.model_checkpoint_dir, exist_ok=True)
        self.loss_plot_dir = f"{self.output_dir}/loss_plots"
        os.makedirs(self.loss_plot_dir, exist_ok=True)
        self.sample_plot_dir = f"{self.output_dir}/sample_plots"
        os.makedirs(self.sample_plot_dir, exist_ok=True)

        self.training_animation_path = f"{self.output_dir}/training_data_animation_{self.test_id}.gif"
        self.generate_training_animation = False if os.path.exists(self.training_animation_path) else True

    @classmethod
    def from_json_file(cls, filepath):
        """Loads configuration settings from a JSON file and returns a Config object."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Config file not found: {filepath}")
        with open(filepath, 'r') as f:
            settings = json.load(f)
        if 'image_size' in settings and isinstance(settings['image_size'], list):
            settings['image_size'] = tuple(settings['image_size'])
        return cls(**settings)

    def to_json_file(self, filepath):
        """Saves the current configuration settings to a JSON file."""
        settings = self.__dict__.copy()
        if isinstance(settings.get('image_size'), tuple):
            settings['image_size'] = list(settings['image_size'])
        if 'device' in settings:
            del settings['device']
        for key in ['model_checkpoint_dir','loss_plot_dir','sample_plot_dir','training_animation_path']:
            if key in settings:
                del settings[key]
        with open(filepath, 'w') as f:
            json.dump(settings, f, indent=4)
        print(f"Configuration saved to {filepath}")