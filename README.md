# masterfiltmodel : Modelling of the 8 "master key filters" in ConvNeXt V2 networks using idealized scale-space filters

Based on the "master key filter hypothesis" proposed by Babaiee et al. (2025), according to which the learned filters in the ConvNeXt architecture can be replaced by a set of 8 "master key filters" with only a marginal decrease in accuracy, we have taken this idea one step further, by replacing the 8 "master key filters" with a set of 8 idealized discrete scale-space filters, and demonstrated that such an abstraction can again be performed with again an only marginal decrease in accuracy.

These results thus show that  the learned filters in the depthwise-separable deep networks based on the ConvNeXt V2 Tiny architecture can be well approximated by discrete scale-space filters. 

In this repository, we provide Python code for the following:

## model-fitting:

computing the parameters of idealized discrete scale-space filters that model the 8 master key filters extracted from the ConvNeXt V2 Tiny architecture

## experiments:

deep networks constructed by replacing the learned filters in the ConvNeXt V2 Tiny architecture with discrete scale-space filters evaluated on the ImageNet dataset

## References:

T. Lindeberg, Z. Babaiee and P. Kiasari (2026) “Modelling and analysis of the 8 filters from
the ’master key filters hypothesis’ for depthwise-separable deep networks in relation to
idealized receptive fields based on scale-space theory”, Journal of Mathematical Imaging and Vision, to appear. 
([preprint](https://doi.org/10.48550/arXiv.2509.12746))

Z. Babaiee, P. Kiasari, D. Rus, and R. Grosu (2025) "The master key filters hypothesis: Deep filters are general", AAAI Conference on Artificial Intelligence. 
([preprint](https://doi.org/10.48550/arXiv.2412.16751)) 

Liu, Z., Mao, H., Wu, C.-Y., Feichtenhofer, C., Darrell, T. & Xie, S. (2022), A ConvNet for the 2020s, in ‘Proc. Computer Vision and Pattern Recognition (CVPR 2022)’, pp. 11976–11986.
([OpenAccess](https://openaccess.thecvf.com/content/CVPR2022/papers/Liu_A_ConvNet_for_the_2020s_CVPR_2022_paper.pdf))

Woo, S., Debnath, S., Hu, R., Chen, X., Liu, Z., Kweon, I. S. & Xie, S. (2023), ConvNeXt V2: Co-designing and scaling ConvNets with masked autoencoders, in ‘Computer Vision and Pattern Recognition (CVPR 2023)’.
([OpenAccess](https://openaccess.thecvf.com/content/CVPR2023/papers/Woo_ConvNeXt_V2_Co-Designing_and_Scaling_ConvNets_With_Masked_Autoencoders_CVPR_2023_paper.pdf))

