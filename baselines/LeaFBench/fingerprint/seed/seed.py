from fingerprint.fingerprint_interface import LLMFingerprintInterface
from fingerprint.seed.utils import get_random_tokens, get_random_embed, get_output_for_tokens, get_output
from fingerprint.seed.correlation_test import test_lineage


class SeedFingerprint(LLMFingerprintInterface):
    """SeedPrint fingerprinting method.

    Supports two input modes:
      - "token" (default): random token IDs through each model's embedding layer.
        Produces model-specific hidden states; recommended for foundation models.
      - "embedding": random continuous embeddings bypassing the embedding layer.
        Recommended for toy models where init→pretrained lineage is tested.

    Default: token input + coset + perdim-only.
    """

    def __init__(self, config=None, accelerator=None):
        super().__init__(config=config, accelerator=accelerator)

    def prepare(self, train_models=None):
        self.input_type = self.config.get('input_type', 'token')
        self.random_input_path = self.config.get('random_input_path', 'cache/seed/fingerprint_input')
        self.num_sequences = self.config.get('num_sequences', 2000)
        self.seq_length = self.config.get('seq_length', 1024)
        self.min_vocab_size = self.config.get('min_vocab_size', 32000)
        self.output_type = self.config.get('output_type', 'hidden_states')
        self.batch_size = self.config.get('batch_size', 4)
        self.ratio_k = self.config.get('ratio_k', 0.10)
        self.normalize = self.config.get('normalize', 'softmax_T10')
        self.identity_mode = self.config.get('identity_mode', 'coset')
        self.use_agg = self.config.get('use_agg', False)

    def get_fingerprint(self, model):
        torch_model, tokenizer = model.load_model()
        if self.input_type == 'token':
            token_path = get_random_tokens(
                self.random_input_path,
                num_sequences=self.num_sequences,
                seq_length=self.seq_length,
                min_vocab_size=self.min_vocab_size,
            )
            fingerprint = get_output_for_tokens(
                torch_model, token_path,
                batch_size=self.batch_size,
                accelerator=self.accelerator,
            )
        else:
            hidden_size = torch_model.config.hidden_size
            emb_path = get_random_embed(
                model.model_path, hidden_size, self.random_input_path,
                num_sequences=self.num_sequences, seq_length=self.seq_length,
            )
            fingerprint = get_output(
                torch_model, emb_path,
                output_type=self.output_type,
                batch_size=self.batch_size,
                accelerator=self.accelerator,
            )
        return fingerprint

    def compare_fingerprints(self, base_model, testing_model):
        base_fingerprint = base_model.get_fingerprint()
        testing_fingerprint = testing_model.get_fingerprint()
        D_base = base_fingerprint.shape[1]
        D_target = testing_fingerprint.shape[1]
        print(f"Base fingerprint shape: {base_fingerprint.shape}, "
              f"Testing: {testing_fingerprint.shape}")

        if D_base != D_target:
            if self.identity_mode == "coset":
                D_min = min(D_base, D_target)
                base_fingerprint = base_fingerprint[:, :D_min]
                testing_fingerprint = testing_fingerprint[:, :D_min]
                print(f"Dimension mismatch, truncating to D={D_min}")
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
            use_agg=self.use_agg,
        )
        print(f"p-value: {p_value:.4g}")

        return 1 - p_value
