from proteinshake.datasets import EnzymeCommissionDataset 
from proteinshake.tasks import ProteinFamilyTask
from proteinshake.tasks import EnzymeClassTask
from proteinshake.tasks import StructuralClassTask
from proteinshake.tasks import GeneOntologyTask



task = EnzymeClassTask(split='structure', split_similarity_threshold=0.7)

print('number of classess', task.num_classes)
dataset = task.dataset
print(dataset)

protein_generator = dataset.proteins(resolution='atom')

print(len(protein_generator))


train_index = task.train_index
print('number of training proteins:', len(train_index))
print(train_index[:10])
valid_index = task.val_index
print('number of validation proteins:', len(valid_index))
test_index = task.test_index
print('number of testing proteins:', len(test_index))


# for protein in protein_generator:
#     # print(protein['protein'].keys())
#     print(protein['protein']['Pfam'])
#     # print('EC Number:', protein['protein']['EC'])
#     # print(protein['protein']['sequence_split_0.7'])
#     break



