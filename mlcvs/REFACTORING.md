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

## GENERAL

[ ] how to handle imports 

## DOC

[X] add nbmake in tests and CI
[ ] tutorialsssss

## IO

[ ] to do? 

## CVS

[X] remove decorators! 
[X] training --> options = self.loss_options.copy()
[ ] add option to add postprocessing after init
[ ] print optim and loss options in summary
[ ] add flag to apply preprocessing during training (default = None, complain when preprocessing is not None)

## UTILS.DATA

[X] Change names to data utils ==> "DictionaryDataset","DictionaryDataModule","FastDictionaryLoader"
[X] Refactor FastDictionaryLoader
[X] add get_stats to datamodule ?
[X] mv from mlcvs.data to mlcvs.data
[X] fix return of dataloaders in 
[ ] Mixed dataset?

## STATS

[ ] Add base class

## TRANSFORM 

[ ] Pairwise distances
[ ] symmetry functions
[ ] Add utils for switching functions? 

## MODELS

[X] NN ==> add batchnorm / dropout / activation function  per layer
[ ] add torch.nn.module as block (load existing module)  

## EXPORT

[ ] check device for normalization buffers 
[ ] add export function?
[ ] export torchscript model as checkpoint



