import os
import sys
import json
import argparse
import logging
import torch
import transformers
from transformers import BertTokenizer, BertForSequenceClassification
import pandas as pd
from collections import Counter



class DataLoader:
    # initialize the class
    def __init__(self, DR):
        # Following is the testing set
        self.df = pd.read_csv('knowledge-neurons/data/pdtb-3.0-easy-fixed.tsv', sep='\t', header=0)
        self.df_pdtb2_implicit = self.df[(self.df['Relation Type'] == 'Implicit') & (self.df['Provenance'].str.contains('PDTB2'))]
        self.df_pdtb2_implicit = self.df_pdtb2_implicit[self.df_pdtb2_implicit['Section'].isin([21, 22])]
        self.df_pdtb2_implicit_test = self.df_pdtb2_implicit.copy()
        # drop index in df_pdtb2_implicit_test
        self.df_pdtb2_implicit_test = self.df_pdtb2_implicit_test.reset_index(drop=True)
        # print the first 5 rows of the dataframe
        print(self.df_pdtb2_implicit_test.head())

        # # Get the training set
        self.df = pd.read_csv('knowledge-neurons/data/pdtb-3.0-easy-fixed.tsv', sep='\t', header=0)
        self.df_pdtb2_implicit = self.df[(self.df['Relation Type'] == 'Implicit') & (self.df['Provenance'].str.contains('PDTB2'))]
        self.df_pdtb2_implicit = self.df_pdtb2_implicit[self.df_pdtb2_implicit['Section'].isin(list(range(2, 20)))]
        self.df_pdtb2_implicit_train = self.df_pdtb2_implicit.copy()
        # drop index in df_pdtb2_implicit_test
        self.df_pdtb2_implicit_train = self.df_pdtb2_implicit_train.reset_index(drop=True)
        # print the first 5 rows of the dataframe
        print(self.df_pdtb2_implicit_train.head())

        # Whether we decide to use testing case. 
        self.use_test = False

        if self.use_test:
            self.df_pdtb2_implicit_train = self.df_pdtb2_implicit_test

        # Get SClass1A for the entire test set, which is a column in the dataframe
        self.SClass1A = self.df_pdtb2_implicit_train['SClass1A'].tolist()

        # Get the Counter of SClass1A
        self.SClass1A_counter = Counter(self.SClass1A)
        # Rank the SClass1A based on the frequency
        self.SClass1A_ranked = self.SClass1A_counter.most_common()
        # print it
        print(json.dumps(self.SClass1A_ranked, indent=4))
        self.DR = DR

        # Get the top 13 SClass1A
        self.top_13_SClass1A = [item[0] for item in self.SClass1A_ranked[:13]]

        print(self.top_13_SClass1A)

        self.big_dict = {} # which is the same as the data structure in 'data_all_allbags.json' in the knowledge neuron paper. 
        self.conn_recoder = {}


class generate_CF_discourse_instance:
    # initialize the class
    def __init__(self, model_name):
        self.hfpath = '/mnt/data/yisong/hf-path'

        # we need to initiate a model, from LLaMA family. 
        self.model = transformers.AutoModelForCausalLM.from_pretrained(model_name, cache_dir=self.hfpath)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, cache_dir=self.hfpath) 
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.CF_Conn = [
            'therefore',
            'because',
            'however',
            'specifically',
            'for example',
            'in addition',
            'in the meantime',
        ]
    
    # def process_input_discourse(self, arg1, arg2, conn, CF_Conn):
    #     # Concatenate arg1 and CF_Conn with a comma and space between them
    #     input_str = '{}, {}'.format(arg1, CF_Conn)
    #     return input_str

    # def process_input_discourse(self, arg1, arg2, conn, CF_Conn):
    #     # not quite work
    #     prompt = (
    #         f"Example:\n"
    #         f"arg1: {arg1}\n"
    #         f"conn: {conn}\n"
    #         f"arg2: {arg2}\n"
    #         f"Now please create a new sentence by replacing '{conn}' with '{CF_Conn}'.\n"
    #         f"Please use some words from arg2, but don't make it the same as arg2.\n"
    #         f"Output: {arg1}, {CF_Conn} "
    #     )
    #     return prompt

    def process_input_discourse(self, arg1, arg2, conn, CF_Conn):
        # not quite work
        prompt = (
            f"arg1: {arg1}\n"
            f"Now please finish the sentence with '{CF_Conn}', make sure the completion is a complete sentence coherent with '{CF_Conn}'.\n"
            f"Output: {arg1}, {CF_Conn},"
        )
        return prompt


    # def process_input_discourse(self, arg1, arg2, conn, CF_Conn):
    #     # Create a concise and context-aware prompt for the model
    #     prompt = (
    #         f"Task:\n"
    #         f"Given the following:\n"
    #         f"- Arg1: \"{arg1}\"\n"
    #         f"- Original Arg2: \"{arg2}\"\n"
    #         f"- Original Connective: \"{conn}\"\n"
    #         f"- New Connective (CF_Conn): \"{CF_Conn}\"\n\n"
    #         f"Complete the string \"Arg1 CF_Conn\" by generating a new Arg2 (CF_Arg2) that aligns with CF_Conn. "
    #         f"Reuse relevant words from the Original Arg2 wherever possible, ensuring coherence with Arg1 and reflecting the meaning of CF_Conn.\n\n"
    #         f"Output:\n"
    #         f"Write the completed string \"Arg1 CF_Conn CF_Arg2.\""
    #     )
    #     return prompt


    
    def generate_CF_discourse(self, arg1, arg2, conn, CF_Conn):
        # Get the processed input string
        input_str = self.process_input_discourse(arg1, arg2, conn, CF_Conn)
        
        # Generate text using the model
        inputs = self.tokenizer(input_str, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs["input_ids"],
                max_new_tokens=50,
                do_sample=True,
                temperature=0.7,
                pad_token_id=self.tokenizer.eos_token_id,
                attention_mask=inputs["attention_mask"]
            )

        # Decode the generated text
        # print('input_str: ', input_str)
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        print('--------------------------------')
        print('generated_text: ', generated_text)
        
        
        return generated_text
        


class CF_PDTB_Generator:
    def __init__(self, model_name):
        # Initialize the discourse generator
        self.generator = generate_CF_discourse_instance(model_name)
        
        # Load PDTB validation data
        self.df = pd.read_csv('knowledge-neurons/data/pdtb-3.0-easy-fixed.tsv', sep='\t', header=0)
        self.df_pdtb2_implicit = self.df[(self.df['Relation Type'] == 'Implicit') & 
                                       (self.df['Provenance'].str.contains('PDTB2'))]
        self.df_pdtb2_implicit_val = self.df_pdtb2_implicit[self.df_pdtb2_implicit['Section'].isin([20])]
        self.df_pdtb2_implicit_val = self.df_pdtb2_implicit_val.reset_index(drop=True)

    def generate_CF_examples(self):
        CF_examples = dict()
        
        # Loop through validation set
        for idx, row in self.df_pdtb2_implicit_val.iterrows():
            arg1 = row['Arg1 SpanList']
            arg2 = row['Arg2 SpanList'] 
            conn = row['Conn1']
            # Get discourse relation
            discourse_relation = row['SClass1A']

            # Create dictionary to store example metadata
            example_dict = {
                'original': {
                    'arg1': arg1,
                    'arg2': arg2,
                    'conn': conn,
                    'relation': discourse_relation
                },
                'counterfactuals': []
            }
            
            # Generate counterfactual examples using each CF connector
            for cf_conn in self.generator.CF_Conn:
                print('idx: ', idx)
                cf_text = self.generator.generate_CF_discourse(arg1, arg2, conn, cf_conn)
                example = {
                    'original_arg1': arg1,
                    'original_arg2': arg2,
                    'original_conn': conn,
                    'cf_conn': cf_conn,
                    'generated_text': cf_text,
                    'original_relation': discourse_relation
                }
                # add the example to the example_dict
                example_dict['counterfactuals'].append(example)
                # add the example_dict to the CF_examples
                CF_examples[idx] = example_dict
            
            # Log progress
            if (idx + 1) % 10 == 0:
                print(f"Processed {idx + 1} examples")
            
            # Save the result once 100 examples are processed or the last example
            if (idx + 1) % 100 == 0 or idx == len(self.df_pdtb2_implicit_val) - 1 or idx == 5:
                save_dir = f"CF_PDTB/data/cf_examples"
                # create the directory if it doesn't exist
                os.makedirs(save_dir, exist_ok=True)
                fp = f"{save_dir}/examples.json"
                with open(fp, 'w') as f:
                    json.dump(CF_examples, f, indent=4)
        return CF_examples


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate discourse relation data')
    parser.add_argument('--DR', type=str, default='Expansion.Conjunction',
                        help='The discourse relation to analyze')
    args = parser.parse_args()

    data_loader = DataLoader(args.DR)
    
    # Initialize CF_PDTB_Generator
    model_name = 'meta-llama/Llama-3.1-8B-Instruct'
    cf_pdtb_generator = CF_PDTB_Generator(model_name)

    cf_examples = cf_pdtb_generator.generate_CF_examples()


