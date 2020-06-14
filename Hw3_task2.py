import sys
import jsonlines

from lama.modules import build_model_by_name
import lama.options as options

def embeddings(args, models, text, text_masked): ###code from get_contextual_embeddings modified
    sentences = [
        [text_masked],  # single-sentence instance
        [text],  # two-sentence
    ]

    #print("Language Models: {}".format(args.models_names))

    for model_name, model in models.items():
        # print("\n{}:".format(model_name))
        contextual_embeddings, sentence_lengths, tokenized_text_list = model.get_contextual_embeddings(
            sentences)

        # contextual_embeddings is a list of tensors, one tensor for each layer.
        # Each element contains one layer of the representations with shape
        # (x, y, z).
        #   x    - the batch size
        #   y    - the sequence length of the batch
        #   z    - the length of each layer vector

        # print(f'Number of layers: {len(contextual_embeddings)}')
        # for layer_id, layer in enumerate(contextual_embeddings):
        #     print(f'Layer {layer_id} has shape: {layer.shape}')
        #
        # print("sentence_lengths: {}".format(sentence_lengths))
        # print("tokenized_text_list: {}".format(tokenized_text_list))

        return contextual_embeddings, tokenized_text_list


###Generate modified args for the lama library (aka imitate the input of a command line)
sys.argv = ['My code for HW3 Task1','--lm', 'bert']
parser = options.get_general_parser()
args = options.parse_args(parser)

###building the model only once (not inside the method for each line)
models = {}
for lm in args.models_names:
    models[lm] = build_model_by_name(lm, args)

###opening the file
with jsonlines.open('./train_testing_output.jsonl') as reader:
    for line in reader.iter():
        dictionary = line

        ###masking the text
        text = dictionary['claim']
        start_masked = dictionary["entity"]['start_character']
        end_masked = dictionary["entity"]['end_character']

        text_masked = text[0:start_masked] + '[MASK]' + text[end_masked:len(text)]

        ### get embeddings
        contextual_embeddings, tokenized_text_list = embeddings(args, models, text_masked, text)
        #print(embeddings)

        ###get index of masked token
        index_of_masked_word = tokenized_text_list[1].index('[MASK]')
        masked_value = tokenized_text_list[1][index_of_masked_word]
        print(masked_value + ' is at index ' + str(index_of_masked_word))

        ###get representations
        Representation_masked = contextual_embeddings[11][1][index_of_masked_word]
        Representation = contextual_embeddings[11][0][index_of_masked_word]
        #print('representation masked ' + str(Representation_masked))






