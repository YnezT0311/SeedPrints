from fingerprint.fingerprint_interface import LLMFingerprintInterface
from fingerprint.seed.get_random_embed import get_random_embed
from fingerprint.seed.get_outputs import get_output
from fingerprint.seed.correlation_test import test_lineage


class SeedFingerprint(LLMFingerprintInterface):
    def __init__(self, config=None, accelerator=None):
        super().__init__(config=config, accelerator=accelerator)

    def prepare(self, train_models=None):
        self.random_input_path = self.config.get('random_input_path', 'cache/seed/fingerprint_input')
        self.num_sequences = self.config.get('num_sequences', 10000)
        self.seq_length = self.config.get('seq_length', 1024)
        self.output_type = self.config.get('output_type', 'hidden_states')
        self.batch_size = self.config.get('batch_size', 32)
        self.ratio_k = self.config.get('ratio_k', 0.05)
        self.normalize = self.config.get('normalize', 'softmax_T10')
        self.identity_mode = self.config.get('identity_mode', 'base')
        self.num_random_baseline_trials = self.config.get('num_random_baseline_trials', 10)

    def get_fingerprint(self, model):
        torch_model, tokenizer = model.load_model()
        hidden_size = torch_model.config.hidden_size
        emb_path = get_random_embed(
            model.model_path, hidden_size, self.random_input_path,
            num_sequences=self.num_sequences, seq_length=self.seq_length
        )
        fingerprint = get_output(
            torch_model, emb_path,
            output_type=self.output_type,
            batch_size=self.batch_size,
            accelerator=self.accelerator
        )
        return fingerprint

    def compare_fingerprints(self, base_model, testing_model):
        base_fingerprint = base_model.get_fingerprint()
        testing_fingerprint = testing_model.get_fingerprint()
        D_base = base_fingerprint.shape[1]
        D_target = testing_fingerprint.shape[1]
        print(f"Base fingerprint shape: {base_fingerprint.shape}, Testing: {testing_fingerprint.shape}")

        if D_base != D_target:
            if self.identity_mode == "coset":
                D_min = min(D_base, D_target)
                base_fingerprint = base_fingerprint[:, :D_min]
                testing_fingerprint = testing_fingerprint[:, :D_min]
                print(f"Dimension mismatch, using coset on shared D={D_min}")
            else:
                print(f"Dimension mismatch ({D_base} vs {D_target}), returning 0.0")
                return 0.0

        buffer_k = int(self.ratio_k * base_fingerprint.shape[1])
        print(f"buffer_k: {buffer_k} (ratio_k={self.ratio_k}, D={base_fingerprint.shape[1]})")

        p_value = test_lineage(
            base_fingerprint, testing_fingerprint,
            buffer_k=buffer_k,
            normalize=self.normalize,
            identity_mode=self.identity_mode,
        )
        print(f"p-value: {p_value:.4g}")

        return 1 - p_value
