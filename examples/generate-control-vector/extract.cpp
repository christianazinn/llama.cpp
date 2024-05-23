#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "llama.h"
#include <string>
#include <vector>
#include <map>

struct DatasetEntry {
    std::string positive;
    std::string negative;

    DatasetEntry(std::string positive, std::string negative) : positive(positive), negative(negative) {}
};

// TODO look at eval-callback.cpp for how to, uh, do eval callback

// TODO sync with gguf.py
enum llm_arch {
    LLM_ARCH_LLAMA,
    LLM_ARCH_FALCON,
    LLM_ARCH_BAICHUAN,
    LLM_ARCH_GROK,
    LLM_ARCH_GPT2,
    LLM_ARCH_GPTJ,
    LLM_ARCH_GPTNEOX,
    LLM_ARCH_MPT,
    LLM_ARCH_STARCODER,
    LLM_ARCH_PERSIMMON,
    LLM_ARCH_REFACT,
    LLM_ARCH_BERT,
    LLM_ARCH_NOMIC_BERT,
    LLM_ARCH_BLOOM,
    LLM_ARCH_STABLELM,
    LLM_ARCH_QWEN,
    LLM_ARCH_QWEN2,
    LLM_ARCH_PHI2,
    LLM_ARCH_PLAMO,
    LLM_ARCH_CODESHELL,
    LLM_ARCH_ORION,
    LLM_ARCH_INTERNLM2,
    LLM_ARCH_MINICPM,
    LLM_ARCH_GEMMA,
    LLM_ARCH_STARCODER2,
    LLM_ARCH_MAMBA,
    LLM_ARCH_XVERSE,
    LLM_ARCH_COMMAND_R,
    LLM_ARCH_UNKNOWN,
};

static const std::map<llm_arch, const char *> LLM_ARCH_NAMES = {
    { LLM_ARCH_LLAMA,           "llama"      },
    { LLM_ARCH_FALCON,          "falcon"     },
    { LLM_ARCH_GROK,            "grok"       },
    { LLM_ARCH_GPT2,            "gpt2"       },
    { LLM_ARCH_GPTJ,            "gptj"       },
    { LLM_ARCH_GPTNEOX,         "gptneox"    },
    { LLM_ARCH_MPT,             "mpt"        },
    { LLM_ARCH_BAICHUAN,        "baichuan"   },
    { LLM_ARCH_STARCODER,       "starcoder"  },
    { LLM_ARCH_PERSIMMON,       "persimmon"  },
    { LLM_ARCH_REFACT,          "refact"     },
    { LLM_ARCH_BERT,            "bert"       },
    { LLM_ARCH_NOMIC_BERT,      "nomic-bert" },
    { LLM_ARCH_BLOOM,           "bloom"      },
    { LLM_ARCH_STABLELM,        "stablelm"   },
    { LLM_ARCH_QWEN,            "qwen"       },
    { LLM_ARCH_QWEN2,           "qwen2"      },
    { LLM_ARCH_PHI2,            "phi2"       },
    { LLM_ARCH_PLAMO,           "plamo"      },
    { LLM_ARCH_CODESHELL,       "codeshell"  },
    { LLM_ARCH_ORION,           "orion"      },
    { LLM_ARCH_INTERNLM2,       "internlm2"  },
    { LLM_ARCH_MINICPM,         "minicpm"    },
    { LLM_ARCH_GEMMA,           "gemma"      },
    { LLM_ARCH_STARCODER2,      "starcoder2" },
    { LLM_ARCH_MAMBA,           "mamba"      },
    { LLM_ARCH_XVERSE,          "xverse"     },
    { LLM_ARCH_COMMAND_R,       "command-r"  },
    { LLM_ARCH_UNKNOWN,         "(unknown)"  },
};

const char* user_tag = "[INST]";
const char* asst_tag = "[/INST]";


std::vector<std::string> get_suffixes(std::string filename) {
    // TODO open file and read suffixes either in JSON or comma separated
    return std::vector<std::string>();
}


std::string make_template(std::string prompt_template, std::string persona, std::string suffix) {
    // TODO how the hell are you going to format this without std::format???
    // TODO each should be formatted "{user_tag} {first template} {persona} {second template} {asst_tag} {suffix}"
    return "TODO";
}


// TODO should you be using a llama_model or llama_model_loader? or something else entirely?
void export_gguf(std::string model_type, std::map<int, ggml_tensor> tensors) {
    // TODO look at gguf_split cpp for inspiration, or try to just do it in bare cpp from the python impl.
    // reimplementing this in C++ will be a pain
    /*
    arch = "controlvector"
    writer = gguf.GGUFWriter(path, arch)
    writer.add_string(f"{arch}.model_hint", self.model_type)
    writer.add_uint32(f"{arch}.layer_count", len(self.directions))
    for layer in self.directions.keys():
        writer.add_tensor(f"direction.{layer}", self.directions[layer])
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    */
}


std::map<int, ggml_tensor> read_representations(const llama_model & model, std::vector<DatasetEntry> inputs, int hidden_layers = 0, int batch_size = 32) {
    // TODO translate
    /*
    if not hidden_layers:
        hidden_layers = range(-1, -model.config.num_hidden_layers, -1)

    # normalize the layer indexes if they're negative
    n_layers = len(model_layer_list(model))
    hidden_layers = [i if i >= 0 else n_layers + i for i in hidden_layers]

    # the order is [positive, negative, positive, negative, ...]
    train_strs = [s for ex in inputs for s in (ex.positive, ex.negative)]

    layer_hiddens = batched_get_hiddens(
        model, tokenizer, train_strs, hidden_layers, batch_size
    )

    # get differences between (positive, negative) pairs
    relative_layer_hiddens = {}
    for layer in hidden_layers:
        relative_layer_hiddens[layer] = (
            layer_hiddens[layer][::2] - layer_hiddens[layer][1::2]
        )

    # get directions for each layer using PCA
    directions: dict[int, np.ndarray] = {}
    for layer in tqdm.tqdm(hidden_layers):
        assert layer_hiddens[layer].shape[0] == len(inputs) * 2

        # fit layer directions
        train = np.vstack(
            relative_layer_hiddens[layer]
            - relative_layer_hiddens[layer].mean(axis=0, keepdims=True)
        )
        pca_model = PCA(n_components=1, whiten=False).fit(train)
        # shape (n_features,)
        directions[layer] = pca_model.components_.astype(np.float32).squeeze(axis=0)

        # calculate sign
        projected_hiddens = project_onto_direction(
            layer_hiddens[layer], directions[layer]
        )

        # order is [positive, negative, positive, negative, ...]
        positive_smaller_mean = np.mean(
            [
                projected_hiddens[i] < projected_hiddens[i + 1]
                for i in range(0, len(inputs) * 2, 2)
            ]
        )
        positive_larger_mean = np.mean(
            [
                projected_hiddens[i] > projected_hiddens[i + 1]
                for i in range(0, len(inputs) * 2, 2)
            ]
        )

        if positive_smaller_mean > positive_larger_mean:  # type: ignore
            directions[layer] *= -1

    return directions
    */
}


// TODO make sure this works because it's copypasted from ChatGPT
template <typename T>
std::vector<std::vector<T>> splitIntoBatches(const std::vector<T>& inputs, int batch_size) {
    std::vector<std::vector<T>> batches;
    int num_batches = (inputs.size() + batch_size - 1) / batch_size; // Calculate the number of batches needed
    for (int p = 0; p < num_batches; ++p) {
        int start = p * batch_size;
        int end = std::min(start + batch_size, static_cast<int>(inputs.size()));
        batches.push_back(std::vector<T>(inputs.begin() + start, inputs.begin() + end));
    }
    return batches;
}


std::map<int, ggml_tensor> batched_get_hiddens(const llama_model & model, std::vector<std::string> inputs, std::vector<int> hidden_layers, int batch_size) {
    
    std::vector<std::vector<std::string>> batched_inputs = splitIntoBatches(inputs, batch_size);
    // TODO I don't actually have a clue what data type this should be 
    std::map<int, std::vector<float>> hidden_states = std::map<int, std::vector<float>>();
    for(int layer : hidden_layers) {
        hidden_states[layer] = std::vector<float>();
    }
    // TODO translate
    /*
    with torch.no_grad():
        for batch in batched_inputs:
            out = model(
                **tokenizer(batch, padding=True, return_tensors="pt").to(model.device),
                output_hidden_states=True,
            )
            for layer in hidden_layers:
                # if not indexing from end, account for embedding hiddens
                hidden_idx = layer + 1 if layer >= 0 else layer
                for batch in out.hidden_states[hidden_idx]:
                    hidden_states[layer].append(batch[-1, :].squeeze().cpu().numpy())
            del out

    return {k: np.vstack(v) for k, v in hidden_states.items()}
    */
    // TODO implement PCA
}


std::vector<float> project_onto_direction(struct ggml_tensor* H, struct ggml_tensor* direction) {
    // TODO translate
    /*
    mag = np.linalg.norm(direction)
    assert not np.isinf(mag)
    return (H @ direction) / mag
    */
}
// translate project_onto_direction but I have no clue what the signature looks like 


int main() {
    // TODO parse args: model, positive and negative personas, suffix filename, template, etc.
    
    // TODO read model somehow
    llama_model * model;
    std::vector<DatasetEntry> dataset;

    std::string filename = "TODO"; // TODO user input filename
    std::vector<std::string> suffixes = get_suffixes(filename);

    // create the dataset
    for(std::string suffix : suffixes) {
        // TODO tokenize the suffix
        std::vector<std::string> tokens; // = tokenizer.tokenize(suffix) but in cpp

        for(int i = 1; i < tokens.size() - 5; i++) {
            std::string truncated; // = tokenizer.convert_tokens_to_string(tokens[:i])

            // TODO zip the personas together
            // TODO support multiple personas ^
            // for loop here

            dataset.push_back(DatasetEntry(make_template(truncated, tokens[i], suffix), make_template(truncated, tokens[i], suffix)));
        }
    }

    std::map<int, ggml_tensor> control_vector = read_representations(*model, dataset);
    export_gguf(LLM_ARCH_NAMES.at(model.arch), control_vector);
}