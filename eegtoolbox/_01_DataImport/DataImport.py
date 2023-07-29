import mne

class DataImport:

    def __init__(self):
        self.folder_path_string = "_00_Datasets/"

    def loadGDFFiles(self, selected_dataset):

        # Select dataset folder
        if selected_dataset == 'BCICIV_2a_Multiple' or selected_dataset == 'BCICIV_2a_Single':
            path = 'BCICIV_2a_gdf'
        else:
            path = 'BCICIV_2a_gdf'

        # Select dataset part
        if selected_dataset == 'BCICIV_2a_Multiple':
            filenames = ["A01T.gdf", 'A02T.gdf', 'A03T.gdf', 'A05T.gdf', 'A06T.gdf', 'A07T.gdf', 'A08T.gdf', 'A09T.gdf']
        elif selected_dataset == 'BCICIV_2a_Single':
            filenames = ["A01T.gdf", 'A02T.gdf']
        else:
            filenames = ["A01T.gdf", 'A02T.gdf']


        self.folder_path_string = self.folder_path_string+path+"/"
        raws = []
        for filename in filenames:
            path = self.folder_path_string + filename
            raws.append(mne.io.read_raw_gdf(path))
        return raws
