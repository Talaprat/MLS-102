from sklearn.model_selection import train_test_split as tts

class improved_model_selection():
    
    def imp_tts(dataset, target, drop=[], **kwargs):
        
        y = dataset[target]
        dataset = dataset.drop(columns=drop)        
        X = dataset.drop(columns=target)
        
        return tts(X, y)