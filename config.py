# --- config.py ---
# This file defines global configuration parameters for the diffusion model.
import torch
import json # Import json module for file operations
import os # Import os for path handling

class Config:
    def __init__(
        self,
        test_id,
        image_size=(128, 128),
        real_data=True, # Added as an argument
        use_salinity=False,
        data_points=500, # This is only relevant for synthetic data

        filepath_t=None, # Default to None, set dynamically if real_data
        filepath_s=None, # Default to None, set dynamically if real_data
        filepath_t_test=None, # Default to None, set dynamically if real_data
        filepath_s_test=None, # Default to None, set dynamically if real_data
        varname_t='sst', # Default value
        varname_s='sss', # Default value

        lat_range=[26,154], # Default value
        lon_range=[120,248], # Default value
        training_day_range=[0,10000,1], # Default value
        T_range=[-2,33], # Default value
        S_range=[32,37], # Default value

        timesteps=1000,
        beta_start=1e-5,
        beta_end=0.01,
        base_unet_channels=32,
        epochs=100,
        batch_size=200,
        learning_rate=1e-5,
        channel_wise_normalization=True,
        validation_split=0.1,
        save_model_after_training=True,

        use_dayofyear_embedding=False,
        dayofyear_embedding_dim=64,
        use_2d_location_embedding=False,
        location_embedding_channels=2,
    
        load_model_for_sampling=False,
        sampling_method='ddpm',
        sample_day=270, # Default value
        sample_day_range=None, # Default to None, set dynamically
        observation_fidelity_weight=1.0,
        observation_samples=1000,
        ):
        
        self.test_id = test_id
        self.image_size = image_size
        self.real_data = real_data
        self.data_points = None if real_data else data_points
        self.use_salinity = use_salinity
        self.channels = 2 if self.use_salinity else 1
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Conditional real data file paths
        if self.real_data:
            # Set default values for file paths if not provided
            self.filepath_t = filepath_t 
            self.filepath_s = filepath_s if self.use_salinity else None
            self.filepath_t_test = filepath_t_test
            self.filepath_s_test = filepath_s_test if self.use_salinity else None
            self.varname_t = varname_t
            self.varname_s = varname_s if self.use_salinity else None
            self.lat_range = lat_range
            self.lon_range = lon_range
            self.training_day_range = training_day_range
            
            self.sample_day = sample_day
            self.sample_day_range = sample_day_range if sample_day_range is not None \
                else [self.sample_day, self.sample_day + 1, 1]
            self.T_range = T_range
            self.S_range = S_range if self.use_salinity else None
        else:
            # Set None for real data paths if not using real data
            self.filepath_t = None
            self.filepath_s = None
            self.filepath_t_test = None
            self.filepath_s_test = None
            self.varname_t = None
            self.varname_s = None
            self.lat_range = None
            self.lon_range = None
            self.training_day_range = None
            self.sample_day = None
            self.sample_day_range = None
            self.T_range = None
            self.S_range = None

        self.output_dir = f"/scratch/cimes/feiyul/Diffusion_OceanDA/{self.test_id}"
        if not os.path.isdir(self.output_dir):
            os.mkdir(self.output_dir)

        self.timesteps = timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.base_unet_channels = base_unet_channels
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.channel_wise_normalization = channel_wise_normalization
        self.validation_split = validation_split
        self.save_model_after_training = save_model_after_training

        self.use_dayofyear_embedding = use_dayofyear_embedding
        self.dayofyear_embedding_dim = dayofyear_embedding_dim
        self.use_2d_location_embedding = use_2d_location_embedding    
        self.location_embedding_channels = location_embedding_channels

        self.load_model_for_sampling = load_model_for_sampling
        self.sampling_method = sampling_method
        self.observation_fidelity_weight = observation_fidelity_weight
        self.observation_samples = observation_samples

        # Dynamic paths must be set after their dependencies are defined
        self.model_checkpoint_path = f"{self.output_dir}/ODA_ch{self.channels}_checkpoint_{self.test_id}.pth"
        self.plot_save_path_t = f"{self.output_dir}/ODA_{self.test_id}_obs{self.observation_samples}_day{self.sample_day}_t.png"
        self.plot_save_path_s = f"{self.output_dir}/ODA_{self.test_id}_obs{self.observation_samples}_day{self.sample_day}_s.png"
        self.loss_plot_save_path = f"{self.output_dir}/loss_plot_{self.test_id}.png"
        self.generate_training_animation = False if self.load_model_for_sampling else True
        self.training_animation_path = f"{self.output_dir}/training_data_animation_{self.test_id}.gif"

    @classmethod
    def from_json_file(cls, filepath):
        """
        Loads configuration settings from a JSON file and returns a Config object.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Config file not found: {filepath}")
        
        with open(filepath, 'r') as f:
            settings = json.load(f)
        
        # Convert list for image_size tuple back to tuple if loaded from JSON
        if 'image_size' in settings and isinstance(settings['image_size'], list):
            settings['image_size'] = tuple(settings['image_size'])

        # Create a Config instance with loaded settings
        # Use **settings to unpack the dictionary into keyword arguments
        return cls(**settings)

    def to_json_file(self, filepath):
        """
        Saves the current configuration settings to a JSON file.
        """
        # Convert tuple to list for JSON serialization if necessary
        settings = self.__dict__.copy()
        if isinstance(settings.get('image_size'), tuple):
            settings['image_size'] = list(settings['image_size'])
        
        # Remove torch.device object as it's not JSON serializable; it's dynamically set.
        if 'device' in settings:
            del settings['device']

        # Remove dynamically generated paths before saving to avoid redundancy,
        # or ensure they are properly handled if they rely on other dynamic vars.
        # For simplicity, we'll exclude them if they are f-strings based on other values.
        # This assumes the dynamic paths are recreated correctly during from_json_file.
        for key in ['model_checkpoint_path', 'plot_save_path', 'loss_plot_save_path', 'training_animation_path']:
            if key in settings:
                del settings[key]
        
        with open(filepath, 'w') as f:
            json.dump(settings, f, indent=4)
        print(f"Configuration saved to {filepath}")

# Create a global config object using the new constructor
# config = Config()