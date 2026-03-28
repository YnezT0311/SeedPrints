from fingerprint.fingerprint_interface import LLMFingerprintInterface
from fingerprint.seed.get_random_embed import get_random_embed
from fingerprint.seed.get_outputs import get_output
from fingerprint.seed.correlation_test import get_corrs
import numpy as np

class SeedFingerprint(LLMFingerprintInterface):
    def __init__(self, config=None, accelerator=None):
        super().__init__(config=config, accelerator=accelerator)

    def prepare(self, train_models=None):
        """
        Prepare the fingerprinting methods.

        Args:
            train_models (optional): Models to train, if necessary.
        """
        self.random_input_path = self.config.get('random_input_path', 'cache/seed/fingerprint_input')
        self.num_sequences = self.config.get('num_sequences', 10000)
        self.seq_length = self.config.get('seq_length', 1024)
        self.output_type = self.config.get('output_type', 'hidden_states')
        self.batch_size = self.config.get('batch_size', 32)
        # self.num_k = self.config.get('num_k', 200)
        self.ratio_k = self.config.get('ratio_k', 0.25)
        self.num_random_baseline_trials = self.config.get('num_random_baseline_trials', 10)
        self.test_type = self.config.get('test_type', 't-test')
    
    def get_fingerprint(self, model):
        """
        Generate a fingerprint for the given text.

        Args:
            text (str): The input text to fingerprint.

        Returns:
            torch.Tensor: The fingerprint tensor.
        """
        torch_model, tokenizer = model.load_model()
        hidden_size = torch_model.config.hidden_size
        # get random embeddings and save to emb_path
        emb_path = get_random_embed(model.model_path, hidden_size, self.random_input_path, num_sequences=self.num_sequences, seq_length=self.seq_length)
        
        # # use accelerator if available
        # if self.accelerator:
        #     torch_model = self.accelerator.prepare(torch_model)
        fingerprint = get_output(
            torch_model, 
            emb_path, 
            output_type=self.output_type, 
            batch_size=self.batch_size, 
            accelerator=self.accelerator
        )
        return fingerprint

    def compare_fingerprints(self, base_model, testing_model):
        """
        Compare two models using their fingerprints.

        Args:
            base_model (ModelInterface): The base model to compare against.
            testing_model (ModelInterface): The model to compare.

        Returns:
            float: Similarity score between the two fingerprints.
        """
        base_fingerprint = base_model.get_fingerprint() # shape [num_samples, hidden_size]
        testing_fingerprint = testing_model.get_fingerprint()
        print(f"Base fingerprint shape: {base_fingerprint.shape}")
        base_fingerprint = base_fingerprint.to(self.accelerator.device)
        testing_fingerprint = testing_fingerprint.to(self.accelerator.device)
        self.num_k = int(self.ratio_k * base_fingerprint.shape[1])

        corr_vector, _, random_corr_pooled = get_corrs(base_fingerprint, testing_fingerprint, self.output_type, self.num_k, self.num_random_baseline_trials)
        if self.test_type == 't-test':
            from scipy.stats import ttest_ind
            stat, p_value = ttest_ind(corr_vector, random_corr_pooled, alternative='greater')
            print(f"One sided t-test: p-value = {p_value}")
        elif self.test_type == 'u-test':
            from scipy.stats import mannwhitneyu
            stat, p_value = mannwhitneyu(corr_vector, random_corr_pooled, alternative='greater')
            print(f"One sided Mann-Whitney U test: p-value = {p_value}")
        else:
            raise ValueError(f"Unknown test type: {self.test_type}")
        
        return 1-p_value