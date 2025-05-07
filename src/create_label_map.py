from src.constants import PREPROCESSDATA_PATH


def create_label_map():
    labels = [
        {'name':'Hello', 'id':1},
        {'name':'Thank you', 'id':2},
        {'name':'Yes', 'id':3},
        {'name':'Ok', 'id':4},
        {'name':'Nice', 'id':5}
    ]

    with open(PREPROCESSDATA_PATH + '/label_map.pbtxt', 'w') as f:
        for label in labels:
            f.write('item {\n')
            f.write(f"\tname:'{label['name']}'\n")
            f.write(f"\tid:{label['id']}\n")
            f.write('}\n')
