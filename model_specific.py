""" module for model specific concretisation of abstract interfaces """
import os
import pandas as pd

from tools.converter import Converter, InputConverter
from tools.inference import Inferencer


class ConcreteConverter(Converter):
    def process_file(self, filepath: str, output_filename: str):
        """ implement for specific model
        expects tsv file with columns:
        classifier: Prediction, Probability_score
        regressor: Prediction"""
        df = pd.read_csv(filepath, sep="\t")
        df.rename(columns={"probability_AMP": "Probability_score", "predicted": "Prediction"})
        df['Prediction'] = df['Prediction'].replace({'AMP': 'AMP', 'nonAMP': 'non-AMP'})
        df.to_csv(output_filename, sep="\t")


class ConcreteInferencer(Inferencer):
    def process_file(self, filepath: str, output_filename: str):
        """ implement for specific model """
        command = f"ampep predict -m pretrained_models/amPEP.model -i {filepath} -o {output_filename}"
        print(command)
        os.system(command)


class ConcreteInputConverter(InputConverter):
    def process_file(self, filepath: str, output_filename: str):
        """ implement for specific model
        base format is fasta"""
        pass
