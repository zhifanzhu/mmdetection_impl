NaiveAssemble does not do normalization inside cuda kernel,
since we can have aff : [B, D^2, H, W], 
we can normalize along vicinity before aff input to AssembleFunciton