nn
    [X] transform
        [X] normSTD
        [X] normRANGE
    [x] models 
        [x] feedforward
        [ ] linear
    [ ] utils
        [ ] lda (sw/sb + fisher)
    [ ] loss
        [ ] mse 
        [ ] lda_eigenvalues
        [ ] fisher_ratio
        [ ] tda_loss
        [ ] tica_eigenvalues
        [ ] autocorrelation

cvs
    [X] supervised
        [X] SimpleCV
    [ ] unsupervised
        [ ] AutoEncoderCV
    [ ] discriminant
        [ ] DeepLDA 
        [ ] DeepTDA
    [ ] slowmodes
        [ ] DeepTICA

utils 
    [ ] io
        [X] load_dataframe
        [ ] dataloader_from_file >> CHANGE IT (1) more general (2) datamodules
    [ ] data
        [X] FastTensorDataloader
        [X] TensorDataModule
        [ ] Mixed datataset (combination of more datasets)
    [ ] fes
        [X] compute_fes >> IMPROVE THE KEYWORDS (e.g. via dict)
