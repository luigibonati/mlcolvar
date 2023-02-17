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


TODO
[ ] change order LightingModule/CVUtils in cvs
[ ] change name of init block defautls / return options in cvs
[ ] move blocks from object to class members 
[ ] remove forward all blocks from cvs
[ ] change training and validation step in single function if possible
[ ] change loss func in kwargs (add kwarsgs also in set_loss_fn) --> check for {} in options

[ ] set optimizer
[ ] add resnet / dropout