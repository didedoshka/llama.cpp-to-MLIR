import logging
import sys

logger = logging.getLogger("reader")

import numpy as np

from gguf import GGUFWriter  # noqa: E402

# Example usage:
def writer_example(gguf_file_path) -> None:
    # Example usage with a file
    gguf_writer = GGUFWriter(gguf_file_path, "didedoshka_test")
    gguf_writer.add_name("didedoshka_test")
    gguf_writer.add_context_length(1)
    gguf_writer.add_embedding_length(2)
    gguf_writer.add_block_count(1)
    gguf_writer.add_tokenizer_model("llama")
    # gguf_writer.add_string("tokenizer.ggml.tokens", "<unk>")
    gguf_writer.add_array("tokenizer.ggml.tokens", "<unk>")
    # gguf_writer.add_token_list(["<", "u", "n", "k", ">"])
    gguf_writer.add_uint32("didedoshka_test.attention.head_count", 1)
    # gguf_writer.add_uint32("didedoshka_test.rope.dimension_count", 48)

    tensor = np.arange(16, dtype=np.float32).reshape((4, 4))

    gguf_writer.add_tensor("token_embd.weight", tensor)
    gguf_writer.add_tensor("output.weight", tensor)
    gguf_writer.add_tensor("blk.0.attn_q.weight", tensor)
    gguf_writer.add_tensor("blk.0.attn_k.weight", tensor)
    gguf_writer.add_tensor("blk.0.attn_v.weight", tensor)

    gguf_writer.write_header_to_file()
    gguf_writer.write_kv_data_to_file()
    gguf_writer.write_tensors_to_file()

    gguf_writer.close()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        logger.info("Usage: writer.py <path_to_gguf_file>")
        sys.exit(1)
    gguf_file_path = sys.argv[1]
    writer_example(gguf_file_path)
