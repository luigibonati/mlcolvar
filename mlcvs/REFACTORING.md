nn
    [X] transform
        [X] normSTD
        [X] normRANGE
    [x] models 
        [x] feedforward
        [ ] linear
    [X] utils
        [X] lda (sw/sb + fisher)
    [X] loss
        [X] mse 
        [X] lda_eigenvalues
        [X] fisher_ratio
        [X] tda_loss
        [X] tica_eigenvalues
        [X] autocorrelation

cvs
    [X] supervised
        [X] SimpleCV
    [X] unsupervised
        [X] AutoEncoderCV
    [X] discriminant
        [X] DeepLDA 
        [X] DeepTDA
    [X] slowmodes
        [X] DeepTICA

utils 
    [ ] io
        [X] load_dataframe
        [ ] dataloader_from_file >> CHANGE IT (1) more general (2) datamodules
    [ ] data
        [X] FastDictionaryLoader
        [X] DictionaryDataModule
        [ ] Mixed datataset (combination of more datasets)
    [ ] fes
        [X] compute_fes >> IMPROVE THE KEYWORDS (e.g. via dict)

TODO
[X] change order LightingModule/CVUtils in cvs
[X] change name of init block defautls / return options in cvs
[X] move blocks from object to class members 
[X] remove forward all blocks from cvs
[X] change training and validation step in single function if possible

[X] change loss func in kwargs (add kwarsgs also in set_loss_fn) --> check for {} in options
[X] MV define_in_features_out_features to initialization of CV class
[X] change name from CV_utils to BaseCV

[X] set optimizer

## IO

[ ] to do?

## UTILS.DATA

[X] Change names to data utils ==> "DictionaryDataset","DictionaryDataModule","FastDictionaryLoader"
[X] Refactor FastDictionaryLoader
[ ] Mixed dataset?

## TRANSFORM 
 
[ ] Pairwise distances
[ ] symmetry functions

## MODELS

[ ] NN ==> add batchnorm / dropout / activation function  per layer
[ ] ADD PRE / POST BLOCKS + options whether to use them 
[ ] add torch.nn.module as block (load existing module)  
[ ] print optim and loss options in summary

## TRAINER

[ ] subclass trainer / use hooks to achieve:
    [ ] on_fit_start --> preprocess dataset + disable pre/post in model
    [ ] on_fit_end --> enable pre/post ?

## EXTRA

[ ] check device for normalization buffers 
[ ] add export function?



