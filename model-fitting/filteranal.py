""" Modelling of 8 filters from "Master key filter hypothesis" in terms of eightidealized receptive fields based on scale-space theory"""


import numpy as np
from math import sqrt, log
from typing import Union
from scipy.signal import correlate
from scipy.ndimage import convolve, correlate1d
from pyscsp.discscsp import dxmask, dymask, dxxmask, dxymask, dyymask, \
                            dxmask3, dymask3, dxxmask3, dxymask3, dyymask3, \
                            filtermean, L1norm, variance, make1Dgaussfilter


def dxmask1d() -> np.array :
    """Defines a (minimal) non-centered mask for discrete approximation of the 
    first-order derivative in the x-direction.
    """
    return np.array([-1/2,  0, +1/2])


def dxpmask1d() -> np.array :
    """Defines a (minimal) non-centered mask for discrete approximation of the 
    first-order derivative in the x-direction.
    """
    return np.array([ 0, -1, +1])


def dxmmask1d() -> np.array :
    """Defines a (minimal) non-centered mask for discrete approximation of the 
    first-order derivative in the x-direction.
    """
    return np.array([-1, +1,  0])


def dxpmask() -> np.array :
    """Defines a (minimal) non-centered mask for discrete approximation of the 
    first-order derivative in the x-direction.
    """
    return np.array([[ 0, -1, +1]])


def dxmmask() -> np.array :
    """Defines a (minimal) non-centered mask for discrete approximation of the 
    first-order derivative in the x-direction.
    """
    return np.array([[-1, +1,  0]])


def dypmask() -> np.array :
    """Defines a (minimal) non-centered mask for discrete approximation of the 
    first-order derivative in the y-direction.
    """
    return np.array([[+1], \
                     [-1], \
                     [ 0]])
                     

def dymmask() -> np.array :
    """Defines a (minimal) non-centered mask for discrete approximation of the 
    first-order derivative in the y-direction.
    """
    return np.array([[ 0], \
                     [+1], \
                     [-1]])


def dxpmask3() -> np.array :
    """Defines a non-centered mask of size 3 x 3 for discrete approximation of the 
    first-order derivative in the x-direction.
    """
    return np.array([[ 0,  0,  0], \
                     [ 0, -1, +1], \
                     [ 0,  0,  0]])


def dxmmask3() -> np.array :
    """Defines a non-centered mask of size 3 x 3 for discrete approximation of the 
    first-order derivative in the x-direction.
    """
    return np.array([[ 0,  0, 0], \
                     [-1, +1, 0], \
                     [ 0,  0, 0]])


def dypmask3() -> np.array :
    """Defines a non-centered mask of size 3 x 3 for discrete approximation of the 
    first-order derivative in the x-direction.
    """
    return np.array([[ 0, +1, 0], \
                     [ 0, -1, 0], \
                     [ 0,  0, 0]])


def dypmask3() -> np.array :
    """Defines a non-centered mask of size 3 x 3 for discrete approximation of the 
    first-order derivative in the x-direction.
    """
    return np.array([[ 0,  0, 0], \
                     [ 0, +1, 0], \
                     [ 0, -1, 0]])

                     
def lapl5mask3() -> np.array :
    """Defines a mask of size 3 x 3 for discrete approximation of the 
    Laplacian operator.
    """
    return np.array([[ 0, +1,  0], \
                     [+1, -4, +1], \
                     [ 0, +1,  0]])


def laplcrossmask3() -> np.array :
    """Defines a mask of size 3 x 3 for discrete approximation of the 
    Laplacian operator.
    """
    return np.array([[+1/2,  0, +1/2], \
                     [   0, -2,     0], \
                     [+1/2,  0, +1/2]])


def laplmixedmask3() -> np.array :
    """Return the most isotropic 3 x 3 discrete approximation of the 
    Laplacian operator.
    """
    return 2/3 * lapl5mask3() + 1/3 * laplcrossmask3()


def keyfilter1() -> np.array :
    """ Master key filter 1 as extracted by Babaiee et al (2025)
    """
    data = [[-0.010366, -0.019109, -0.014336, -0.004956, -0.007191, -0.017836, -0.014678], \
            [-0.018185, -0.023806, -0.003971, 0.003164, -0.009877, -0.019598, -0.012438], \
            [-0.009972, -0.016889, 0.006929, -0.112378, -0.004454, -0.014606, -0.011968], \
            [-0.029286, -0.049522, -0.093870, -0.230421, -0.060529, -0.050651, -0.028252], \
            [-0.029460, -0.062816, 0.018601, 0.936427, 0.037004, -0.056659, -0.028216], \
            [-0.024469, -0.016398, 0.004305, 0.115283, 0.008598, -0.018598, -0.018682], \
            [-0.021804, -0.016337, 0.005643, 0.086478, 0.000976, -0.018258, -0.022568]]

    return np.array(data)


def keyfilter2() -> np.array :
    """ Master key filter 2 as extracted by Babaiee et al (2025)
    """
    filt = [[-0.002871, -0.006335, -0.023576, -0.049731, -0.038692, -0.019982, -0.001609], \
            [-0.015574, -0.017483, -0.024908, -0.044207, -0.032561, -0.022291, -0.028825], \
            [-0.016807, -0.004999, -0.012284, -0.055665, 0.064783, -0.010828, -0.012208], \
            [0.001370, 0.038502, -0.059240, -0.456545, 0.848257, 0.129801, 0.069732], \
            [0.001566, 0.013770, 0.005058, -0.121033, 0.074078, 0.020350, 0.007190], \
            [-0.009344, -0.014567, -0.011630, -0.050635, -0.025819, -0.010569, -0.012529], \
            [0.002350, -0.006887, -0.012085, -0.042283, 0.000708, -0.007356, 0.004438]]

    return np.array(filt)          


def keyfilter3() -> np.array :
    """ Master key filter 3 as extracted by Babaiee et al (2025)
    """
    filt = [[-0.025760, -0.021993, -0.015623, 0.068278, -0.020230, -0.026704, -0.029474], \
            [-0.029693, -0.020450, 0.008347, 0.140394, 0.013333, -0.023140, -0.033615], \
            [-0.025873, -0.041326, 0.098596, 0.878125, 0.108857, -0.053538, -0.035708], \
            [-0.023234, -0.024152, -0.076522, -0.355209, -0.090919, -0.026583, -0.030537], \
            [-0.018112, -0.003603, -0.045245, -0.144964, -0.045179, -0.010070, -0.017247], \
            [-0.011080, -0.009287, 0.008988, 0.009857, 0.002208, -0.010719, -0.012225], \
                [-0.005171, 0.001564, 0.003216, 0.013518, 0.007650, 0.001186, -0.000934]]
                
    return np.array(filt)    


def keyfilter4() -> np.array :
    """ Master key filter 4 as extracted by Babaiee et al (2025)
    """
    filt = [[-0.041307, -0.030676, -0.024261, -0.007210, 0.001897, -0.001310, -0.007319], \
            [-0.036686, -0.011045, -0.037788, -0.006021, 0.028528, 0.012414, -0.012684], \
            [-0.006411, 0.000642, 0.033764, -0.047998, 0.004343, 0.024774, 0.009462], \
            [0.039193, 0.078214, 0.869568, -0.350899, -0.303579, -0.000505, -0.001577], \
            [-0.015323, 0.001697, 0.050901, -0.013813, -0.049879, -0.002864, -0.000469], \
            [-0.034810, -0.006612, -0.005195, 0.000068, 0.003687, 0.000717, -0.015397], \
            [-0.037408, -0.024825, -0.012339, -0.007305, -0.000221, -0.003765, -0.002368]]

    return np.array(filt)    


def keyfilter5() -> np.array :
    """ Master key filter 5 as extracted by Babaiee et al (2025)
    """
    filt = [[0.050752, 0.018862, 0.039537, 0.006379, -0.044870, -0.024716, -0.073547], \
            [0.036470, 0.026162, 0.045467, 0.016569, -0.021388, -0.013378, -0.065535], \
            [0.096410, 0.086563, 0.188319, 0.017527, -0.169982, -0.062293, -0.094769], \
            [0.198740, 0.202768, 0.542363, -0.033217, -0.528883, -0.198141, -0.215364], \
            [0.087319, 0.081274, 0.189242, 0.013053, -0.217298, -0.087242, -0.105545], \
            [0.041305, 0.031175, 0.066061, 0.014364, -0.035550, -0.018613, -0.065024], \
            [0.053047, 0.022884, 0.046441, -0.000353, -0.043196, -0.028261, -0.071890]] 
   
    return np.array(filt)  


def keyfilter6() -> np.array :
    """ Master key filter 6 as extracted by Babaiee et al (2025)
    """
    filt = [[-0.070076, -0.047989, -0.076243, -0.156696, -0.072218, -0.040500, -0.062643], \
            [-0.028031, -0.005619, -0.060923, -0.144135, -0.035828, 0.002474, -0.031041], \
            [-0.029497, -0.036818, -0.215620, -0.468564, -0.219833, -0.034334, -0.043523], \
            [-0.012474, -0.007903, 0.009237, 0.019302, 0.008470, -0.000447, 0.000818], \
            [0.023980, 0.033675, 0.201968, 0.676272, 0.200072, 0.021273, 0.031519], \
            [-0.003614, 0.017934, 0.061616, 0.159518, 0.046287, 0.014208, 0.005265], \
            [0.021008, 0.026337, 0.049661, 0.144345, 0.058595, 0.031835, 0.038904]] 

    return np.array(filt)


def keyfilter7() -> np.array :
    """ Master key filter 7 as extracted by Babaiee et al (2025)
    """
    filt = [[-0.005660, -0.009521, -0.013251, -0.022730, -0.016777, -0.003587, -0.007026], \
            [-0.010241, -0.004743, -0.015582, -0.047566, -0.014404, -0.002725, -0.012811], \
            [-0.008007, -0.006216, -0.043614, -0.064494, -0.045387, -0.009789, -0.012796], \
            [-0.011767, -0.030798, -0.014348, 0.982706, -0.022891, -0.037026, -0.017894], \
            [-0.012432, -0.013301, -0.053900, -0.074982, -0.057881, -0.017230, -0.018007], \
            [-0.011493, -0.006475, -0.012195, -0.047276, -0.012390, -0.003429, -0.013235], \
            [-0.010339, -0.012779, -0.021011, -0.033894, -0.016922, -0.008399, -0.013484]] 

    return np.array(filt)
  

def keyfilter8() -> np.array :
    """ Master key filter 8 as extracted by Babaiee et al (2025)
    """
    filt = [[-0.039747, -0.037104, -0.036442, -0.015018, -0.038648, -0.028850, -0.035683], \
            [-0.039553, -0.032229, -0.043352, -0.035793, -0.040037, -0.028182, -0.037951], \
            [-0.035896, -0.040227, -0.018008, 0.156419, -0.011734, -0.039894, -0.035016], \
            [-0.017596, -0.038753, 0.160821, 0.920885, 0.152043, -0.037856, -0.015814], \
            [-0.040770, -0.046278, -0.027596, 0.147155, -0.030494, -0.047664, -0.040714], \
            [-0.040421, -0.033754, -0.040347, -0.036521, -0.038895, -0.029883, -0.040003], \
            [-0.042854, -0.039693, -0.043615, -0.023412, -0.037513, -0.034438, -0.043075]]

    return np.array(filt)


def keyfilter(
    idx : Union[int, str]
    ) -> np.array :
  """ Returns the "master key filter" with index idx as extracted by Babaiee et al (2025)
  """
  # The original "master key filters" extracted by Babaiee et al
  if idx == 1:
    return keyfilter1()
  if idx == 2:
    return keyfilter2()
  if idx == 3:
    return keyfilter3()
  if idx == 4:
    return keyfilter4()
  if idx == 5:
    return keyfilter5()
  if idx == 6:
    return keyfilter6()    
  if idx == 7:
    return keyfilter7()  
  if idx == 8:
    return keyfilter8()

  # Normalized versions of the "master key filters"
  if idx == 'norm1':
    return keyfilter1()/keyfiltmonomresponse(1, 0, 1)
  if idx == 'norm2':
    return keyfilter2()/keyfiltmonomresponse(2, 1, 0)
  if idx == 'norm3':
    return keyfilter3()/keyfiltmonomresponse(3, 0, 1)
  if idx == 'norm4':
    return keyfilter4()/keyfiltmonomresponse(4, 1, 0)
  if idx == 'norm5':
    return keyfilter5()/keyfiltmonomresponse(5, 1, 0)
  if idx == 'norm6':
    return keyfilter6()/keyfiltmonomresponse(6, 0, 1)

  if idx == 'prelnorm7':
    return keyfilter7() - Chat(7)
  if idx == 'prelnorm8':
    return keyfilter8() - Chat(8)
  
  if idx == 'norm7':
    return keyfilter('prelnorm7')/keyfiltmonomresponse('prelnorm7', 0, 0)
  if idx == 'norm8':
    return keyfilter('prelnorm8')/keyfiltmonomresponse('prelnorm8', 0, 0)  

  # Idealized models based on scale estimates computed from discrete variances
  # combined with from variances of continuous Gaussian derivative kernels,
  # with separate scale parameters in horizontal and vertical directions
  if idx == 'sepfromcontvarnorm1':
    sigmax = keyfiltersigmahat(1, 'x', 'sepfromcontvar')
    sigmay = keyfiltersigmahat(1, 'y', 'sepfromcontvar')
    return keyfiltertempl(1, sigmax, sigmay)
  
  if idx == 'sepfromcontvarnorm2':
    sigmax = keyfiltersigmahat(2, 'x', 'sepfromcontvar')
    sigmay = keyfiltersigmahat(2, 'y', 'sepfromcontvar')
    return keyfiltertempl(2, sigmax, sigmay)

  if idx == 'sepfromcontvarnorm3':
    sigmax = keyfiltersigmahat(3, 'x', 'sepfromcontvar')
    sigmay = keyfiltersigmahat(3, 'y', 'sepfromcontvar')
    return keyfiltertempl(3, sigmax, sigmay)

  if idx == 'sepfromcontvarnorm4':
    sigmax = keyfiltersigmahat(4, 'x', 'sepfromcontvar')
    sigmay = keyfiltersigmahat(4, 'y', 'sepfromcontvar')
    return keyfiltertempl(4, sigmax, sigmay)

  if idx == 'sepfromcontvarnorm5':
    sigmax = keyfiltersigmahat(5, 'x', 'sepfromcontvar')
    sigmay = keyfiltersigmahat(5, 'y', 'sepfromcontvar')
    return keyfiltertempl(5, sigmax, sigmay)

  if idx == 'sepfromcontvarnorm6':
    sigmax = keyfiltersigmahat(6, 'x', 'sepfromcontvar')
    sigmay = keyfiltersigmahat(6, 'y', 'sepfromcontvar')
    return keyfiltertempl(6, sigmax, sigmay)

  if idx == 'sepfromcontvarnorm7':
    # Use estimate from two-parameter l2-based method
    sigma = keyfiltersigmahat(7, 'x', 'l2diffsamescale')
    alpha = keyfilter7alphahat('l2diffsamescale')
    return keyfiltertempl7(sigma, alpha)
  
  # if idx == 'sepfromcontvarnorm7':
  #   sigmax = keyfiltersigmahat(7, 'x', 'sepfromcontvar')
  #   sigmay = keyfiltersigmahat(7, 'y', 'sepfromcontvar')
  #   return keyfiltertempl(7, sigmax, sigmay)

  if idx == 'sepfromcontvarnorm8':
    sigmax = keyfiltersigmahat(8, 'x', 'sepfromcontvar')
    sigmay = keyfiltersigmahat(8, 'y', 'sepfromcontvar')
    return keyfiltertempl(8, sigmax, sigmay)

  # Idealized models based on scale estimates computed from discrete variances
  # combined with from variances of continuous Gaussian derivative kernels,
  # with similar scale parameters in horizontal and vertical directions
  # as well as within certain subgroups (12348, 56, 7) of the filters
  if idx == 'jointfromcontvarnorm1':
    sigmax = keyfiltersigmahat(1, 'x', 'jointfromcontvar')
    sigmay = keyfiltersigmahat(1, 'y', 'jointfromcontvar')
    return keyfiltertempl(1, sigmax, sigmay)

  if idx == 'jointfromcontvarnorm2':
    sigmax = keyfiltersigmahat(2, 'x', 'jointfromcontvar')
    sigmay = keyfiltersigmahat(2, 'y', 'jointfromcontvar')
    return keyfiltertempl(2, sigmax, sigmay)

  if idx == 'jointfromcontvarnorm3':
    sigmax = keyfiltersigmahat(3, 'x', 'jointfromcontvar')
    sigmay = keyfiltersigmahat(3, 'y', 'jointfromcontvar')
    return keyfiltertempl(3, sigmax, sigmay)

  if idx == 'jointfromcontvarnorm4':
    sigmax = keyfiltersigmahat(4, 'x', 'jointfromcontvar')
    sigmay = keyfiltersigmahat(4, 'y', 'jointfromcontvar')
    return keyfiltertempl(4, sigmax, sigmay)

  if idx == 'jointfromcontvarnorm5':
    sigmax = keyfiltersigmahat(5, 'x', 'jointfromcontvar')
    sigmay = keyfiltersigmahat(5, 'y', 'jointfromcontvar')
    return keyfiltertempl(5, sigmax, sigmay)

  if idx == 'jointfromcontvarnorm6':
    sigmax = keyfiltersigmahat(6, 'x', 'jointfromcontvar')
    sigmay = keyfiltersigmahat(6, 'y', 'jointfromcontvar')
    return keyfiltertempl(6, sigmax, sigmay)

  if idx == 'jointfromcontvarnorm7':
    # Use estimate from two-parameter l2-based method
    sigma = keyfiltersigmahat(7, 'x', 'l2diffsamescale')
    alpha = keyfilter7alphahat('l2diffsamescale')
    return keyfiltertempl7(sigma, alpha)
  
  # if idx == 'jointfromcontvarnorm7':
  #   sigmax = keyfiltersigmahat(7, 'x', 'jointfromcontvar')
  #   sigmay = keyfiltersigmahat(7, 'y', 'jointfromcontvar')
  #   return keyfiltertempl(7, sigmax, sigmay)

  if idx == 'jointfromcontvarnorm8':
    sigmax = keyfiltersigmahat(8, 'x', 'jointfromcontvar')
    sigmay = keyfiltersigmahat(8, 'y', 'jointfromcontvar')
    return keyfiltertempl(8, sigmax, sigmay)

  # Idealized models based on scale estimates computed from discrete weighted variances
  # of idealized models of receptive fields matched to corresponding discrete
  # weighted variances of the corresponding "master key filters"
  if idx == 'sepfromdiscvarnorm1':
    sigmax = keyfiltersigmahat(1, 'x', 'sepfromdiscvar')
    sigmay = keyfiltersigmahat(1, 'y', 'sepfromdiscvar')
    return keyfiltertempl(1, sigmax, sigmay)

  if idx == 'sepfromdiscvarnorm2':
    sigmax = keyfiltersigmahat(2, 'x', 'sepfromdiscvar')
    sigmay = keyfiltersigmahat(2, 'y', 'sepfromdiscvar')
    return keyfiltertempl(2, sigmax, sigmay)

  if idx == 'sepfromdiscvarnorm3':
    sigmax = keyfiltersigmahat(3, 'x', 'sepfromdiscvar')
    sigmay = keyfiltersigmahat(3, 'y', 'sepfromdiscvar')
    return keyfiltertempl(3, sigmax, sigmay)

  if idx == 'sepfromdiscvarnorm4':
    sigmax = keyfiltersigmahat(4, 'x', 'sepfromdiscvar')
    sigmay = keyfiltersigmahat(4, 'y', 'sepfromdiscvar')
    return keyfiltertempl(4, sigmax, sigmay)

  if idx == 'sepfromdiscvarnorm5':
    sigmax = keyfiltersigmahat(5, 'x', 'sepfromdiscvar')
    sigmay = keyfiltersigmahat(5, 'y', 'sepfromdiscvar')
    return keyfiltertempl(5, sigmax, sigmay)

  if idx == 'sepfromdiscvarnorm6':
    sigmax = keyfiltersigmahat(6, 'x', 'sepfromdiscvar')
    sigmay = keyfiltersigmahat(6, 'y', 'sepfromdiscvar')
    return keyfiltertempl(6, sigmax, sigmay)

  if idx == 'sepfromdiscvarnorm7':
      # Use estimate from two-parameter l2-based method
      sigma = keyfiltersigmahat(7, 'x', 'l2diffsamescale')
      alpha = keyfilter7alphahat('l2diffsamescale')
      return keyfiltertempl7(sigma, alpha)
  
  # if idx == 'sepfromdiscvarnorm7':
  #   sigmax = keyfiltersigmahat(7, 'x', 'sepfromdiscvar')
  #   sigmay = keyfiltersigmahat(7, 'y', 'sepfromdiscvar')
  #   return keyfiltertempl(7, sigmax, sigmay)

  if idx == 'sepfromdiscvarnorm8':
    sigmax = keyfiltersigmahat(8, 'x', 'sepfromdiscvar')
    sigmay = keyfiltersigmahat(8, 'y', 'sepfromdiscvar')
    return keyfiltertempl(8, sigmax, sigmay)

  # Idealized models based on scale estimates computed from discrete weighted variances
  # of idealized models of receptive fields matched to corresponding discrete
  # weighted variances of the corresponding "master key filters",
  # combined with from variances of continuous Gaussian derivative kernels,
  # with similar scale parameters in horizontal and vertical directions
  # as well as within certain subgroups (12348, 56, 7) of the filters
  if idx == 'jointfromdiscvarnorm1':
    sigmax = keyfiltersigmahat(1, 'x', 'jointfromdiscvar')
    sigmay = keyfiltersigmahat(1, 'y', 'jointfromdiscvar')
    return keyfiltertempl(1, sigmax, sigmay)

  if idx == 'jointfromdiscvarnorm2':
    sigmax = keyfiltersigmahat(2, 'x', 'jointfromdiscvar')
    sigmay = keyfiltersigmahat(2, 'y', 'jointfromdiscvar')
    return keyfiltertempl(2, sigmax, sigmay)

  if idx == 'jointfromdiscvarnorm3':
    sigmax = keyfiltersigmahat(3, 'x', 'jointfromdiscvar')
    sigmay = keyfiltersigmahat(3, 'y', 'jointfromdiscvar')
    return keyfiltertempl(3, sigmax, sigmay)

  if idx == 'jointfromdiscvarnorm4':
    sigmax = keyfiltersigmahat(4, 'x', 'jointfromdiscvar')
    sigmay = keyfiltersigmahat(4, 'y', 'jointfromdiscvar')
    return keyfiltertempl(4, sigmax, sigmay)

  if idx == 'jointfromdiscvarnorm5':
    sigmax = keyfiltersigmahat(5, 'x', 'jointfromdiscvar')
    sigmay = keyfiltersigmahat(5, 'y', 'jointfromdiscvar')
    return keyfiltertempl(5, sigmax, sigmay)

  if idx == 'jointfromdiscvarnorm6':
    sigmax = keyfiltersigmahat(6, 'x', 'jointfromdiscvar')
    sigmay = keyfiltersigmahat(6, 'y', 'jointfromdiscvar')
    return keyfiltertempl(6, sigmax, sigmay)

  if idx == 'jointfromdiscvarnorm7':
    # Use estimate from two-parameter l2-based method
    sigma = keyfiltersigmahat(7, 'x', 'l2diffsamescale')
    alpha = keyfilter7alphahat('l2diffsamescale')
    return keyfiltertempl7(sigma, alpha)
  
  # if idx == 'jointfromdiscvarnorm7':
  #   sigmax = keyfiltersigmahat(7, 'x', 'jointfromdiscvar')
  #   sigmay = keyfiltersigmahat(7, 'y', 'jointfromdiscvar')
  #   return keyfiltertempl(7, sigmax, sigmay)

  if idx == 'jointfromdiscvarnorm8':
    sigmax = keyfiltersigmahat(8, 'x', 'jointfromdiscvar')
    sigmay = keyfiltersigmahat(8, 'y', 'jointfromdiscvar')
    return keyfiltertempl(8, sigmax, sigmay)

  # Idealized models based on scale estimates computed from minimization of
  # the discrete l1-norm between the idealized receptive field models and
  # the normalized versions of the learned filters, using different values
  # of the scale parameters in the horizontal and vertical directions.
  if idx == 'sepscalefroml1diff1':
    sigmax = keyfiltersigmahat(1, 'x', 'l1diffsepscale')
    sigmay = keyfiltersigmahat(1, 'y', 'l1diffsepscale')
    return keyfiltertempl(1, sigmax, sigmay)
  if idx == 'sepscalefroml1diff2':
    sigmax = keyfiltersigmahat(2, 'x', 'l1diffsepscale')
    sigmay = keyfiltersigmahat(2, 'y', 'l1diffsepscale')
    return keyfiltertempl(2, sigmax, sigmay)
  if idx == 'sepscalefroml1diff3':
    sigmax = keyfiltersigmahat(3, 'x', 'l1diffsepscale')
    sigmay = keyfiltersigmahat(3, 'y', 'l1diffsepscale')
    return keyfiltertempl(3, sigmax, sigmay)
  if idx == 'sepscalefroml1diff4':
    sigmax = keyfiltersigmahat(4, 'x', 'l1diffsepscale')
    sigmay = keyfiltersigmahat(4, 'y', 'l1diffsepscale')
    return keyfiltertempl(4, sigmax, sigmay)
  if idx == 'sepscalefroml1diff5':
    sigmax = keyfiltersigmahat(5, 'x', 'l1diffsepscale')
    sigmay = keyfiltersigmahat(5, 'y', 'l1diffsepscale')
    return keyfiltertempl(5, sigmax, sigmay)
  if idx == 'sepscalefroml1diff6':
    sigmax = keyfiltersigmahat(6, 'x', 'l1diffsepscale')
    sigmay = keyfiltersigmahat(6, 'y', 'l1diffsepscale')
    return keyfiltertempl(6, sigmax, sigmay)
  if idx == 'sepscalefroml1diff7':
    # Use estimate from two-parameter l1-based method
    sigma = keyfiltersigmahat(7, 'x', 'l1diffsamescale')
    alpha = keyfilter7alphahat('l1diffsamescale')
    return keyfiltertempl7(sigma, alpha)
  # if idx == 'sepscalefroml1diff7':
  #   sigmax = keyfiltersigmahat(7, 'x', 'l1diffsepscale')
  #   sigmay = keyfiltersigmahat(7, 'y', 'l1diffsepscale')
  #   return keyfiltertempl(7, sigmax, sigmay)
  if idx == 'sepscalefroml1diff8':
    sigmax = keyfiltersigmahat(8, 'x', 'l1diffsepscale')
    sigmay = keyfiltersigmahat(8, 'y', 'l1diffsepscale')
    return keyfiltertempl(8, sigmax, sigmay)
  
  # Idealized models based on scale estimates computed from minimization of
  # the discrete l1-norm between the idealized receptive field models and
  # the normalized versions of the learned filters, using the same values
  # of the scale parameters in the horizontal and vertical directions.
  if idx == 'samescalefroml1diff1':
    sigmax = keyfiltersigmahat(1, 'x', 'l1diffsamescale')
    sigmay = keyfiltersigmahat(1, 'y', 'l1diffsamescale')
    return keyfiltertempl(1, sigmax, sigmay)
  if idx == 'samescalefroml1diff2':
    sigmax = keyfiltersigmahat(2, 'x', 'l1diffsamescale')
    sigmay = keyfiltersigmahat(2, 'y', 'l1diffsamescale')
    return keyfiltertempl(2, sigmax, sigmay)
  if idx == 'samescalefroml1diff3':
    sigmax = keyfiltersigmahat(3, 'x', 'l1diffsamescale')
    sigmay = keyfiltersigmahat(3, 'y', 'l1diffsamescale')
    return keyfiltertempl(3, sigmax, sigmay)
  if idx == 'samescalefroml1diff4':
    sigmax = keyfiltersigmahat(4, 'x', 'l1diffsamescale')
    sigmay = keyfiltersigmahat(4, 'y', 'l1diffsamescale')
    return keyfiltertempl(4, sigmax, sigmay)
  if idx == 'samescalefroml1diff5':
    sigmax = keyfiltersigmahat(5, 'x', 'l1diffsamescale')
    sigmay = keyfiltersigmahat(5, 'y', 'l1diffsamescale')
    return keyfiltertempl(5, sigmax, sigmay)
  if idx == 'samescalefroml1diff6':
    sigmax = keyfiltersigmahat(6, 'x', 'l1diffsamescale')
    sigmay = keyfiltersigmahat(6, 'y', 'l1diffsamescale')
    return keyfiltertempl(6, sigmax, sigmay)
  if idx == 'samescalefroml1diff7':
    # Use estimate from two-parameter l1-based method
    sigma = keyfiltersigmahat(7, 'x', 'l1diffsamescale')
    alpha = keyfilter7alphahat('l1diffsamescale')
    return keyfiltertempl7(sigma, alpha)
  # if idx == 'samescalefroml1diff7':
  #   sigmax = keyfiltersigmahat(7, 'x', 'l1diffsamescale')
  #   sigmay = keyfiltersigmahat(7, 'y', 'l1diffsamescale')
  #   return keyfiltertempl(7, sigmax, sigmay)
  if idx == 'samescalefroml1diff8':
    sigmax = keyfiltersigmahat(8, 'x', 'l1diffsamescale')
    sigmay = keyfiltersigmahat(8, 'y', 'l1diffsamescale')
    return keyfiltertempl(8, sigmax, sigmay)

  # Idealized models based on scale estimates computed from minimization of
  # the discrete l1-norm between the idealized receptive field models and
  # the normalized versions of the learned filters, combined with pooling
  # over the groups {1, 2, 3, 4, 8} and {5, 6}
  if idx == 'jointscalefroml1diff1':
    sigmax = keyfiltersigmahat(1, 'x', 'l1diffjointscale')
    sigmay = keyfiltersigmahat(1, 'y', 'l1diffjointscale')
    return keyfiltertempl(1, sigmax, sigmay)
  if idx == 'jointscalefroml1diff2':
    sigmax = keyfiltersigmahat(2, 'x', 'l1diffjointscale')
    sigmay = keyfiltersigmahat(2, 'y', 'l1diffjointscale')
    return keyfiltertempl(2, sigmax, sigmay)
  if idx == 'jointscalefroml1diff3':
    sigmax = keyfiltersigmahat(3, 'x', 'l1diffjointscale')
    sigmay = keyfiltersigmahat(3, 'y', 'l1diffjointscale')
    return keyfiltertempl(3, sigmax, sigmay)
  if idx == 'jointscalefroml1diff4':
    sigmax = keyfiltersigmahat(4, 'x', 'l1diffjointscale')
    sigmay = keyfiltersigmahat(4, 'y', 'l1diffjointscale')
    return keyfiltertempl(4, sigmax, sigmay)
  if idx == 'jointscalefroml1diff5':
    sigmax = keyfiltersigmahat(5, 'x', 'l1diffjointscale')
    sigmay = keyfiltersigmahat(5, 'y', 'l1diffjointscale')
    return keyfiltertempl(5, sigmax, sigmay)
  if idx == 'jointscalefroml1diff6':
    sigmax = keyfiltersigmahat(6, 'x', 'l1diffjointscale')
    sigmay = keyfiltersigmahat(6, 'y', 'l1diffjointscale')
    return keyfiltertempl(6, sigmax, sigmay)
  if idx == 'jointscalefroml1diff7':
    # Use estimate from two-parameter l1-based method
    sigma = keyfiltersigmahat(7, 'x', 'l1diffsamescale')
    alpha = keyfilter7alphahat('l1diffsamescale')
    return keyfiltertempl7(sigma, alpha)
  # if idx == 'jointscalefroml1diff7':
  #   sigmax = keyfiltersigmahat(7, 'x', 'l1diffjointscale')
  #   sigmay = keyfiltersigmahat(7, 'y', 'l1diffjointscale')
  #   return keyfiltertempl(7, sigmax, sigmay)
  if idx == 'jointscalefroml1diff8':
    sigmax = keyfiltersigmahat(8, 'x', 'l1diffjointscale')
    sigmay = keyfiltersigmahat(8, 'y', 'l1diffjointscale')
    return keyfiltertempl(8, sigmax, sigmay)

  # Idealized models based on scale estimates computed from minimization of
  # the discrete l2-norm between the idealized receptive field models and
  # the normalized versions of the learned filters, using different values
  # of the scale parameters in the horizontal and vertical directions.
  if idx == 'sepscalefroml2diff1':
    sigmax = keyfiltersigmahat(1, 'x', 'l2diffsepscale')
    sigmay = keyfiltersigmahat(1, 'y', 'l2diffsepscale')
    return keyfiltertempl(1, sigmax, sigmay)
  if idx == 'sepscalefroml2diff2':
    sigmax = keyfiltersigmahat(2, 'x', 'l2diffsepscale')
    sigmay = keyfiltersigmahat(2, 'y', 'l2diffsepscale')
    return keyfiltertempl(2, sigmax, sigmay)
  if idx == 'sepscalefroml2diff3':
    sigmax = keyfiltersigmahat(3, 'x', 'l2diffsepscale')
    sigmay = keyfiltersigmahat(3, 'y', 'l2diffsepscale')
    return keyfiltertempl(3, sigmax, sigmay)
  if idx == 'sepscalefroml2diff4':
    sigmax = keyfiltersigmahat(4, 'x', 'l2diffsepscale')
    sigmay = keyfiltersigmahat(4, 'y', 'l2diffsepscale')
    return keyfiltertempl(4, sigmax, sigmay)
  if idx == 'sepscalefroml2diff5':
    sigmax = keyfiltersigmahat(5, 'x', 'l2diffsepscale')
    sigmay = keyfiltersigmahat(5, 'y', 'l2diffsepscale')
    return keyfiltertempl(5, sigmax, sigmay)
  if idx == 'sepscalefroml2diff6':
    sigmax = keyfiltersigmahat(6, 'x', 'l2diffsepscale')
    sigmay = keyfiltersigmahat(6, 'y', 'l2diffsepscale')
    return keyfiltertempl(6, sigmax, sigmay)
  if idx == 'sepscalefroml2diff7':
    # Use estimate from two-parameter l1-based method
    sigma = keyfiltersigmahat(7, 'x', 'l2diffsamescale')
    alpha = keyfilter7alphahat('l2diffsamescale')
    return keyfiltertempl7(sigma, alpha)
  # if idx == 'sepscalefroml2diff7':
  #   sigmax = keyfiltersigmahat(7, 'x', 'l2diffsepscale')
  #   sigmay = keyfiltersigmahat(7, 'y', 'l2diffsepscale')
  #  return keyfiltertempl(7, sigmax, sigmay)
  if idx == 'sepscalefroml2diff8':
    sigmax = keyfiltersigmahat(8, 'x', 'l2diffsepscale')
    sigmay = keyfiltersigmahat(8, 'y', 'l2diffsepscale')
    return keyfiltertempl(8, sigmax, sigmay)

  # Idealized models based on scale estimates computed from minimization of
  # the discrete l2-norm between the idealized receptive field models and
  # the normalized versions of the learned filters, using the same values
  # of the scale paraemters in the horizontal and the vertical directions.
  if idx == 'samescalefroml2diff1':
    sigmax = keyfiltersigmahat(1, 'x', 'l2diffsamescale')
    sigmay = keyfiltersigmahat(1, 'y', 'l2diffsamescale')
    return keyfiltertempl(1, sigmax, sigmay)
  if idx == 'samescalefroml2diff2':
    sigmax = keyfiltersigmahat(2, 'x', 'l2diffsamescale')
    sigmay = keyfiltersigmahat(2, 'y', 'l2diffsamescale')
    return keyfiltertempl(2, sigmax, sigmay)
  if idx == 'samescalefroml2diff3':
    sigmax = keyfiltersigmahat(3, 'x', 'l2diffsamescale')
    sigmay = keyfiltersigmahat(3, 'y', 'l2diffsamescale')
    return keyfiltertempl(3, sigmax, sigmay)
  if idx == 'samescalefroml2diff4':
    sigmax = keyfiltersigmahat(4, 'x', 'l2diffsamescale')
    sigmay = keyfiltersigmahat(4, 'y', 'l2diffsamescale')
    return keyfiltertempl(4, sigmax, sigmay)
  if idx == 'samescalefroml2diff5':
    sigmax = keyfiltersigmahat(5, 'x', 'l2diffsamescale')
    sigmay = keyfiltersigmahat(5, 'y', 'l2diffsamescale')
    return keyfiltertempl(5, sigmax, sigmay)
  if idx == 'samescalefroml2diff6':
    sigmax = keyfiltersigmahat(6, 'x', 'l2diffsamescale')
    sigmay = keyfiltersigmahat(6, 'y', 'l2diffsamescale')
    return keyfiltertempl(6, sigmax, sigmay)
  if idx == 'samescalefroml2diff7':
    # Use estimate from two-parameter l2-based method
    sigma = keyfiltersigmahat(7, 'x', 'l2diffsamescale')
    alpha = keyfilter7alphahat('l2diffsamescale')
    return keyfiltertempl7(sigma, alpha)
  # if idx == 'samescalefroml2diff7':
  #   sigmax = keyfiltersigmahat(7, 'x', 'l2diffsamescale')
  #   sigmay = keyfiltersigmahat(7, 'y', 'l2diffsamescale')
  #   return keyfiltertempl(7, sigmax, sigmay)
  if idx == 'samescalefroml2diff8':
    sigmax = keyfiltersigmahat(8, 'x', 'l2diffsamescale')
    sigmay = keyfiltersigmahat(8, 'y', 'l2diffsamescale')
    return keyfiltertempl(8, sigmax, sigmay)

  # Idealized models based on scale estimates computed from minimization of
  # the discrete l2-norm between the idealized receptive field models and
  # the normalized versions of the learned filters, combined with pooling
  # over the groups {1, 2, 3, 4, 8} and {5, 6}
  if idx == 'jointscalefroml2diff1':
    sigmax = keyfiltersigmahat(1, 'x', 'l2diffjointscale')
    sigmay = keyfiltersigmahat(1, 'y', 'l2diffjointscale')
    return keyfiltertempl(1, sigmax, sigmay)
  if idx == 'jointscalefroml2diff2':
    sigmax = keyfiltersigmahat(2, 'x', 'l2diffjointscale')
    sigmay = keyfiltersigmahat(2, 'y', 'l2diffjointscale')
    return keyfiltertempl(2, sigmax, sigmay)
  if idx == 'jointscalefroml2diff3':
    sigmax = keyfiltersigmahat(3, 'x', 'l2diffjointscale')
    sigmay = keyfiltersigmahat(3, 'y', 'l2diffjointscale')
    return keyfiltertempl(3, sigmax, sigmay)
  if idx == 'jointscalefroml2diff4':
    sigmax = keyfiltersigmahat(4, 'x', 'l2diffjointscale')
    sigmay = keyfiltersigmahat(4, 'y', 'l2diffjointscale')
    return keyfiltertempl(4, sigmax, sigmay)
  if idx == 'jointscalefroml2diff5':
    sigmax = keyfiltersigmahat(5, 'x', 'l2diffjointscale')
    sigmay = keyfiltersigmahat(5, 'y', 'l2diffjointscale')
    return keyfiltertempl(5, sigmax, sigmay)
  if idx == 'jointscalefroml2diff6':
    sigmax = keyfiltersigmahat(6, 'x', 'l2diffjointscale')
    sigmay = keyfiltersigmahat(6, 'y', 'l2diffjointscale')
    return keyfiltertempl(6, sigmax, sigmay)
  if idx == 'jointscalefroml2diff7':
    # Use estimate from two-parameter l2-based method
    sigma = keyfiltersigmahat(7, 'x', 'l2diffsamescale')
    alpha = keyfilter7alphahat('l2diffsamescale')
    return keyfiltertempl7(sigma, alpha)  
  # if idx == 'jointscalefroml2diff7':
  #   sigmax = keyfiltersigmahat(7, 'x', 'l2diffjointscale')
  #   sigmay = keyfiltersigmahat(7, 'y', 'l2diffjointscale')
  #   return keyfiltertempl(7, sigmax, sigmay)
  if idx == 'jointscalefroml2diff8':
    sigmax = keyfiltersigmahat(8, 'x', 'l2diffjointscale')
    sigmay = keyfiltersigmahat(8, 'y', 'l2diffjointscale')
    return keyfiltertempl(8, sigmax, sigmay)

  raise ValueError(f'The index ({idx}) must be a value between 1 and 8 combined with one of the given string specifiers')


def shiftmonomial1d(
        order : int,
        N : int,
        offset : float
    ) -> np.array :
    """Generate an image that represents a monomial f(x) = (x - offset)^order, of 
    size 2*N + 1, and with the origin at the center.
    """
    x = np.linspace(-N, N, 2*N+1) - offset

    return x**order


def shiftmonomial2d(
        xorder : int,
        yorder : int,
        N : int,
        xoffset : float = 0.0,
        yoffset : float = 0.0
    ) -> np.array :
    """Computes the values of the polynomial (x - xoffset)^m (y - yoffset)^n 
    for m = xorder and n = yorder on a 2-D centered coordinate grid with 
    unit grid spacing
    """
    xgrid = shiftmonomial1d(xorder, N, xoffset)
    ygrid = np.flip(shiftmonomial1d(yorder, N, yoffset))
                        
    return np.outer(ygrid, xgrid)


def keyfiltadjmonomial2d(
        idx : int,
        xorder : int,
        yorder : int,
        N : int
    ) -> np.array :        
    """Computes the values of the polynomial (x - xoffset)^m (y - yoffset)^n 
    for m = xorder and n = yorder on a 2-D centered coordinate grid with 
    unit grid spacing, with the offset vector (xoffset, yoffset) set to 
    spatial mean value (mx, my) of the "master key filter" with index idx.
    """
    filtermeanabs = filtermean(np.abs(keyfilter(idx)))

    # In the filtermean function in the pyscsp package, the y-direction
    # goes in the upward direction
    mx = filtermeanabs[0]
    my = filtermeanabs[1]

    return shiftmonomial2d(xorder, yorder, N, mx, my)


def keyfiltmonomresponse(
        idx : int,
        xorder : int,
        yorder : int
    ) -> float :  
    """Computes the response of the "master key filter" with index idx
    to the shift-adjusted monomial of order (xorder, yorder) at the center
    and with the shift-adjustment of the monomial determined from the
    mean absolute value of the filter"""

    # The "master key filters are of size 7x7 with 7 = 2*N + 1
    N = 3
    
    adjmonomial = keyfiltadjmonomial2d(idx, xorder, yorder, N)
    monomresponse = convolve(keyfilter(idx), adjmonomial)

    return monomresponse[N, N]


def keyfilterDCcomp(
        idx : int,
        C : float
    ) -> np.array :
    """Subtracts the DC component C from the key filter with index idx
    """
    return keyfilter(idx) - C


def sqrtdetvarkeyfilterDCcomp(
        idx : int,
        C : float
    ) -> float :
    """Computes the determinant of the variance-based spatial spread 
    measure for the DC-compensated key filter with index idx
    """
    var = variance(np.abs(keyfilterDCcomp(idx, C)))

    return sqrt(var[0, 0] * var[1, 1] - var[0, 1] * var[1, 0])


def mapsqrtdetvarkeyfilterDCcomp(
        idx : int,
        Cmin : float,
        Cmax : float,
        numsamples : int
    ) -> (np.array, np.array) :
    """Computes the determinant of the variance-based spread measure for
    a set of DC-compensated versions of the key filter
    """
    Cvec = np.linspace(Cmin, Cmax, numsamples)
    varvec = np.zeros(len(Cvec))

    for i, C in np.ndenumerate(Cvec):
        varvec[i] = sqrtdetvarkeyfilterDCcomp(idx, C)

    return varvec, Cvec


def findminval1d(
        magnvec : np.array,
        Cvec : np.array
    ) -> (float, int) :
    """Determines the minimum value in the 1-D array magnvec, given a parameterization
    of that domain according to the 1-D array Cvec, and interpolates that quantized 
    value to finer resolution using parabolic interpolation
    """
    minpt = np.argmin(magnvec)

    if minpt in (0, len(magnvec) - 1):
        return magnvec[closestidx]

    minval, _ = interpolparextr(Cvec[minpt - 1], Cvec[minpt], Cvec[minpt + 1], \
                                magnvec[minpt - 1], magnvec[minpt], magnvec[minpt + 1])

    return minval


def interpolparextr(
        t1 : float,
        t2 : float,
        t3 : float,
        f1 : float,
        f2 : float,
        f3 : float
    ) -> (float, float) :
    """ Interpolates a one-dimensional parabola through the points (t1, f1), 
    (t2, f2), (t3, f3) and determines the position tpos and the value fextr of 
    the local extremum of the interpolating function. Provided that the point 
    (t2, f2) is a strict local extremum of f, it is guaranteed that t1 <= tpos <= t3.

    Note! This version assumes a constant spacing in t such that t2 - t1 = t3 - t2
    """
    # Determine the parameters of the interpolating parabola
    # g(u) = A u^2/2 + B u + C = 0, based on a unit spacing
    a = f1 - 2*f2 + f3
    b = (f3 - f1) / 2
    c = f2

    # Determine the position of the interpolating extremum
    dx = -b / a
    fextr = c - b**2 / (2 * a)

    # Interpolate the t-value
    if dx > 0:
        tpos = t2 + dx * (t3 - t2)
    else:
        tpos = t2 + dx * (t2 - t1)

    return tpos, fextr


def Chat(idx : int) -> float :
    """Returns an estimate of the DC correction for key filters 7 and 8
    These numerical estimates have been computed by external computations 
    in an enclosed Jupyter Notebook
    """
    if (idx == 7) :
        return -0.011775472973773075
    if (idx == 8) :
        return -0.0386329733224079

    raise ValueError(f'The index ({idx}) must be either 7 or 8')


def idealfilter1d(
        kind : str,      
        sigma : float,
        N : int,
        scspmethod : str = 'discgauss',
        epsilon : float = 0.00000001
    ) -> np.array :
    """Computes an 1-D component of an ideal receptiva field of either,
    corresponding to the application of a difference operator of type
    kind in {'I', 'delta', 'deltap', 'deltam'} to a discrete scale-space
    kernel of type scspmethod at scale sigma.
    """
    scspfilter = make1Dgaussfilter(sigma, scspmethod, epsilon)

    length = len(scspfilter)
    if  length > 2 * N + 1:
        # Extract the central 2*N + 1 elements assuming odd size 
        skip = round((length - 2 * N - 1)/2)
        smoothfilter = scspfilter[skip : skip + 2 * N + 1]
    elif length < 2 * N + 1:
        # Extend the size to 2 * N + 1
        pad = round((2 * N + 1 - length)/2)
        smoothfilter = np.pad(scspfilter, (pad, pad))
    else:
        smoothfilter = scspfilter

    # Apply the specified difference operator to the filter
    if kind == 'I':
        return smoothfilter
    if kind == 'delta':
        return correlate1d(smoothfilter, dxmask1d())
    if kind == 'deltap':
        return correlate1d(smoothfilter, dxpmask1d())
    if kind == 'deltam':
        return correlate1d(smoothfilter, dxmmask1d())

    raise ValueError(f'Unknown kind ({kind}) of difference operator')
        

def idealfilterkind(
        idx : Union[int, str],
        coord : 'str'
    ) -> str :
    """Returns the type of 1-D difference operator that is to be used in
    an ideal scale-space model of the 'master key filter' with index idx
    along the coordinate direction coord in {'x', 'y'}
    """
    if coord == 'x':
        if idx == 'norm1':
            return 'I'
        if idx == 'norm2':
            return 'deltam'
        if idx == 'norm3':
            return 'I'
        if idx == 'norm4':
            return 'deltap'
        if idx == 'norm5':
            return 'delta'
        if idx == 'norm6':
            return 'I'
        if idx == 'norm7':
            return 'I'
        if idx == 'norm8':
            return 'I'

        raise ValueError(f'The argument idx({idx}) must be between norm1 ... norm8')

    elif coord == 'y':
        if idx == 'norm1':
            return 'deltap'
        if idx == 'norm2':
            return 'I'
        if idx == 'norm3':
            return 'deltam'
        if idx == 'norm4':
            return 'I'
        if idx == 'norm5':
            return 'I'
        if idx == 'norm6':
            return 'delta'
        if idx == 'norm7':
            return 'I'
        if idx == 'norm8':
            return 'I'

        raise ValueError(f'The argument idx({idx}) must be between norm1 ... norm8')

    else:
        raise ValueError(f'The argument coord({coord}) must be either x or y')


def idealsepfilter2d(
        kind : str,   
        sigmax : float,
        sigmay : float,
        N : int,
        scspmethod : str = 'discgauss',
        epsilon : float = 0.00000001
    ) -> np.array :
    """Computes an idealized receptive field of size N x N pixels and of type
    kind in {'I', 'dx', 'dxp', 'dxm', 'dy', 'dyp', 'dym'} with scale parameters
    sigmax and sigmay in the horizontal and vertical coordinate directions, respectively,
    using the scale-space discretization method for defining the discrete scale-space filter.
    """
    if kind == 'I':
        xfilter = idealfilter1d('I', sigmax, N, scspmethod, epsilon)
        yfilter = idealfilter1d('I', sigmay, N, scspmethod, epsilon)        
        return np.outer(yfilter, xfilter)

    if kind == 'dx':
        xfilter = idealfilter1d('delta', sigmax, N, scspmethod, epsilon)
        yfilter = idealfilter1d('I', sigmay, N, scspmethod, epsilon)        
        return np.outer(yfilter, xfilter)

    if kind == 'dxp':
        xfilter = idealfilter1d('deltap', sigmax, N, scspmethod, epsilon)
        yfilter = idealfilter1d('I', sigmay, N, scspmethod, epsilon)        
        return np.outer(yfilter, xfilter)

    if kind == 'dxm':
        xfilter = idealfilter1d('deltam', sigmax, N, scspmethod, epsilon)
        yfilter = idealfilter1d('I', sigmay, N, scspmethod, epsilon)        
        return np.outer(yfilter, xfilter)

    if kind == 'dy':
        xfilter = idealfilter1d('I', sigmax, N, scspmethod, epsilon)
        # By convention the vertical direction is directed upwards
        # Therefore, the sign of the filter is reversed
        yfilter = -idealfilter1d('delta', sigmay, N, scspmethod, epsilon)        
        return np.outer(yfilter, xfilter)

    if kind == 'dyp':
        xfilter = idealfilter1d('I', sigmax, N, scspmethod, epsilon)
        # By convention the vertical direction is directed upwards
        # Therefore, 'deltap' is replaced by 'deltam' and the sign is reversed
        yfilter = -idealfilter1d('deltam', sigmay, N, scspmethod, epsilon)        
        return np.outer(yfilter, xfilter)

    if kind == 'dym':
        xfilter = idealfilter1d('I', sigmax, N, scspmethod, epsilon)
        # By convention the vertical direction is directed upwards
        # Therefore, 'deltam' is replaced by 'deltap' and the sign is reversed
        yfilter = -idealfilter1d('deltap', sigmay, N, scspmethod, epsilon)        
        return np.outer(yfilter, xfilter)

    raise ValueError(f'The argument kind ({kind}) must be in the set \
(I, dx, dxp, dxm, dy, dyp, dym)')


def keyfiltertempl(
        idx : int,
        sigmax : float,
        sigmay : float,
        scspmethod : str = 'discgauss',
        epsilon : float = 0.00000001
    ) -> np.array :
    """Returns an idealized template filter of a structurally similar shape as 
    the 'master key filter' with index idx, given values of the spatial scale
    parameters sigmax and sigmay in the horizontal and vertical directions, respectively.

    Note, however, that the sign conventions of these idealized filters are similar 
    to those of corresponding Gaussian derivative filters, and thereby deviating
    from the qualitative signs of Filters 2 and 3 among the 'master key filters'.
    """
    N = 3

    if idx == 1:
        return idealsepfilter2d('dyp', sigmax, sigmay, N, scspmethod, epsilon)
    if idx == 2:
        return idealsepfilter2d('dxm', sigmax, sigmay, N, scspmethod, epsilon)
    if idx == 3:
        return idealsepfilter2d('dym', sigmax, sigmay, N, scspmethod, epsilon)
    if idx == 4:
        return idealsepfilter2d('dxp', sigmax, sigmay, N, scspmethod, epsilon)
    if idx == 5:
        return idealsepfilter2d('dx', sigmax, sigmay, N, scspmethod, epsilon)
    if idx == 6:
        return idealsepfilter2d('dy', sigmax, sigmay, N, scspmethod, epsilon)
    if idx == 7:
        return idealsepfilter2d('I', sigmax, sigmay, N, scspmethod, epsilon)
    if idx == 8:
        return idealsepfilter2d('I', sigmax, sigmay, N, scspmethod, epsilon)

    raise ValueError(f'The index ({idx}) of the filter must between 1 and 8')


def keyfiltertempl7(
        sigma : float,
        alpha : float,
        scspmethod : str = 'discgauss',
        epsilon : float = 0.00000001
    ) -> np.array :
    """Returns an idealized template filter of a structurally similar shape as 
    the 'master key filter' with index 7, given values of the spatial scale
    parameter sigmax = sigmay = sigma in the horizontal and vertical directions, 
    respectively, and the parameter alpha.
    """
    N = 3
    idfilter = idealsepfilter2d('I', 0.0, 0.0, N, scspmethod, epsilon)
    smoothfilter = idealsepfilter2d('I', sigma, sigma, N, scspmethod, epsilon)
    laplgaussfilter = correlate(smoothfilter, lapl5mask3(), 'same')

    return idfilter - alpha * laplgaussfilter


def weightfiltermean(
        spatfilter : np.ndarray,
        weightfilter : np.array
    ) -> (float, float) :
    """Computes the weighted spatial mean vector of a 2-D filter, assumed to
    be non-negative, given a weighting filter, also assumed to be non-negative.
    """
    return filtermean(np.multiply(spatfilter, weightfilter))

    
def weightvariance(
        spatfilter : np.array,
        weightfilter : np.array
    ) -> np.ndarray :
    """Computes the weighted spatial variance matrix of a 2-D filter, assumed to
    be non-negative, given a weighting filter, also assumed to be non-negative.
    """
    return variance(np.multiply(spatfilter, weightfilter))


def keyfiltweightfiltermean(
    idx : int,
    sigmax : float,
    sigmay : float,
    scspmethod : str = 'discgauss',
    epsilon : float = 0.00000001
    ) -> (float, float) :
    """Computes the weighted spatial mean of the 'master key filter' with
    index idx, assuming initial estimates of the spatial scale parameters
    sigmax and sigmay.'
    """
    weightfilter = keyfiltertempl(idx, sigmax, sigmay, scspmethod, epsilon)

    if idx == 1:
        return weightfiltermean(np.abs(keyfilter('norm1')), \
                                np.abs(keyfiltertempl(idx, sigmax, sigmay, \
                                                      scspmethod, epsilon)))
    if idx == 2:
        return weightfiltermean(np.abs(keyfilter('norm2')), \
                                np.abs(keyfiltertempl(idx, sigmax, sigmay, \
                                                      scspmethod, epsilon)))
    if idx == 3:
        return weightfiltermean(np.abs(keyfilter('norm3')), \
                                np.abs(keyfiltertempl(idx, sigmax, sigmay, \
                                                      scspmethod, epsilon)))
    if idx == 4:
        return weightfiltermean(np.abs(keyfilter('norm4')), \
                                np.abs(keyfiltertempl(idx, sigmax, sigmay, \
                                                      scspmethod, epsilon)))
    if idx == 5:
        return weightfiltermean(np.abs(keyfilter('norm5')), \
                                np.abs(keyfiltertempl(idx, sigmax, sigmay, \
                                                      scspmethod, epsilon)))
    if idx == 6:
        return weightfiltermean(np.abs(keyfilter('norm6')), \
                                np.abs(keyfiltertempl(idx, sigmax, sigmay, \
                                                      scspmethod, epsilon)))
    if idx == 7:
        return weightfiltermean(np.abs(keyfilter('norm7')), \
                                np.abs(keyfiltertempl(idx, sigmax, sigmay, \
                                                      scspmethod, epsilon)))
    if idx == 8:
        return weightfiltermean(np.abs(keyfilter('norm8')), \
                                np.abs(keyfiltertempl(idx, sigmax, sigmay, \
                                                      scspmethod, epsilon)))

    raise ValueError(f'The index ({idx}) of the filter must between 1 and 8')


def keyfiltweightvariance(
    idx : int,
    sigmax : float,
    sigmay : float,
    scspmethod : str = 'discgauss',
    epsilon : float = 0.00000001
    ) -> np.array :
    """Computes the weighted spatial variance of the 'master key filter' with
    index idx, assuming initial estimates of the spatial scale parameters
    sigmax and sigmay.'
    """
    weightfilter = keyfiltertempl(idx, sigmax, sigmay, scspmethod, epsilon)

    if idx == 1:
        return weightvariance(np.abs(keyfilter('norm1')), \
                              np.abs(keyfiltertempl(idx, sigmax, sigmay, \
                                                    scspmethod, epsilon)))
    if idx == 2:
        return weightvariance(np.abs(keyfilter('norm2')), \
                              np.abs(keyfiltertempl(idx, sigmax, sigmay, \
                                                    scspmethod, epsilon)))
    if idx == 3:
        return weightvariance(np.abs(keyfilter('norm3')), \
                              np.abs(keyfiltertempl(idx, sigmax, sigmay, \
                                                    scspmethod, epsilon)))
    if idx == 4:
        return weightvariance(np.abs(keyfilter('norm4')), \
                              np.abs(keyfiltertempl(idx, sigmax, sigmay, \
                                                    scspmethod, epsilon)))
    if idx == 5:
        return weightvariance(np.abs(keyfilter('norm5')), \
                              np.abs(keyfiltertempl(idx, sigmax, sigmay, \
                                                    scspmethod, epsilon)))
    if idx == 6:
        return weightvariance(np.abs(keyfilter('norm6')), \
                              np.abs(keyfiltertempl(idx, sigmax, sigmay, \
                                                    scspmethod, epsilon)))
    if idx == 7:
        return weightvariance(np.abs(keyfilter('norm7')), \
                              np.abs(keyfiltertempl(idx, sigmax, sigmay, \
                                                    scspmethod, epsilon)))
    if idx == 8:
        return weightvariance(np.abs(keyfilter('norm8')), \
                              np.abs(keyfiltertempl(idx, sigmax, sigmay, \
                                                    scspmethod, epsilon)))
    else:
        raise ValueError(f'The index ({idx}) of the filter must between 1 and 8')
    


def keyfiltersigmahat(
        idx : int,
        coord : 'str',
        sigmamethod : 'str',
        scspmethod : str = 'discgauss',
        epsilon : float = 0.00000001
    ) -> float :
    """Computes a scale estimate for an idealized model of the
    'master key filter' with index idx for the coordinate coord
    using different methods for computing the estimates"""
    
    if sigmamethod == 'sepfromcontvar':
        if idx == 7:
            sigmax = 1/sqrt(2.0)
            sigmay = 1/sqrt(2.0)
        else:
            sigmax = 1.0
            sigmay = 1.0
        varmat = keyfiltweightvariance(idx, sigmax, sigmay, \
                                       scspmethod, epsilon)

        if idx == 1 and coord == 'x':
            return sqrt(2 * varmat[0][0])           
        if idx == 1 and coord == 'y':
            return sqrt(2 * varmat[1][1] / 3)           
        if idx == 2 and coord == 'x':
            return sqrt(2 * varmat[0][0] / 3)           
        if idx == 2 and coord == 'y':
            return sqrt(2 * varmat[1][1])  
        if idx == 3 and coord == 'x':
            return sqrt(2 * varmat[0][0])           
        if idx == 3 and coord == 'y':
            return sqrt(2 * varmat[1][1] / 3)
        if idx == 4 and coord == 'x':
            return sqrt(2 * varmat[0][0] / 3)           
        if idx == 4 and coord == 'y':
            return sqrt(2 * varmat[1][1])

        if idx == 5 and coord == 'x':
            return sqrt(2 * varmat[0][0] / 3)           
        if idx == 5 and coord == 'y':
            return sqrt(2 * varmat[1][1])
        if idx == 6 and coord == 'x':
            return sqrt(2 * varmat[0][0])           
        if idx == 6 and coord == 'y':
            return sqrt(2 * varmat[1][1] / 3)

        if idx == 7 and coord == 'x':
            # Use result from two-parameter l2-method
            return keyfiltersigmahat(idx, 'x', 'l2diffsamescale')
        if idx == 7 and coord == 'y':
            # Use result from two-parameter l2-method
            return keyfiltersigmahat(idx, 'y', 'l2diffsamescale')

        # if idx == 7 and coord == 'x':
        #     return sqrt(2 * varmat[0][0])           
        # if idx == 7 and coord == 'y':
        #     return sqrt(2 * varmat[1][1])
        
        if idx == 8 and coord == 'x':
            return sqrt(2 * varmat[0][0])           
        if idx == 8 and coord == 'y':
            return sqrt(2 * varmat[1][1])
        
        raise ValueError(f'The index ({idx}) must be between 1 and 8')

    if sigmamethod == "jointfromcontvar":
        if idx == 1 or idx == 2 or idx == 3 or idx == 4 or idx == 8:

            sigma1 = \
              keyfiltersigmahat(1, 'x', 'sepfromcontvar', scspmethod) * \
              keyfiltersigmahat(1, 'y', 'sepfromcontvar', scspmethod)
            sigma2 = \
              keyfiltersigmahat(2, 'x', 'sepfromcontvar', scspmethod) * \
              keyfiltersigmahat(2, 'y', 'sepfromcontvar', scspmethod)
            sigma3 = \
              keyfiltersigmahat(3, 'x', 'sepfromcontvar', scspmethod) * \
              keyfiltersigmahat(3, 'y', 'sepfromcontvar', scspmethod)
            sigma4 = \
              keyfiltersigmahat(4, 'x', 'sepfromcontvar', scspmethod) * \
              keyfiltersigmahat(4, 'y', 'sepfromcontvar', scspmethod)
            sigma8 = \
              keyfiltersigmahat(8, 'x', 'sepfromcontvar', scspmethod) * \
              keyfiltersigmahat(8, 'y', 'sepfromcontvar', scspmethod)

            return (sigma1 * sigma2 * sigma3 * sigma4 * sigma8) ** (1/10)

        if idx == 5 or idx == 6:
            sigma5 = \
              keyfiltersigmahat(5, 'x', 'sepfromcontvar', scspmethod) * \
              keyfiltersigmahat(5, 'y', 'sepfromcontvar', scspmethod)
            sigma6 = \
              keyfiltersigmahat(6, 'x', 'sepfromcontvar', scspmethod) * \
              keyfiltersigmahat(6, 'y', 'sepfromcontvar', scspmethod)
      
            return (sigma5 * sigma6) ** (1/4)

        if idx == 7:
            sigma7 = \
              keyfiltersigmahat(7, 'x', 'sepfromcontvar', scspmethod) * \
              keyfiltersigmahat(7, 'y', 'sepfromcontvar', scspmethod)

            return sigma7 ** (1/2)

        raise ValueError(f'The filter index ({idx}) must be between 1 and 8')

    if sigmamethod == 'sepfromdiscvar':
        if idx == 1 and coord == 'x':
            # (xvardiff1, sigmaxvec1) = mapkeyfiltxvardiff(1, 0.64, 0.65, 100)
            # return findminval1d(xvardiff1**2, sigmaxvec1)
            return 0.644467181641027
        if idx == 1 and coord == 'y':
            # (yvardiff1, sigmayvec1) = mapkeyfiltyvardiff(1, 0.57, 0.59, 100)
            # return findminval1d(yvardiff1**2, sigmayvec1)
            return 0.5828832090268337
        if idx == 2 and coord == 'x':
            # (xvardiff2, sigmaxvec2) = mapkeyfiltxvardiff(2, 0.58, 0.59, 100)
            # return findminval1d(xvardiff2**2, sigmaxvec2) 
            return 0.5857839906147607
        if idx == 2 and coord == 'y':
            # (yvardiff2, sigmayvec2) = mapkeyfiltyvardiff(2, 0.64, 0.65, 100)
            # return filteranal(yvardiff2**2, sigmayvec2)
            return 0.6439526929438278
        if idx == 3 and coord == 'x':
            # (xvardiff3, sigmaxvec3) = mapkeyfiltxvardiff(3, 0.68, 0.7, 100)
            # return findminval1d(xvardiff3**2, sigmaxvec3)
            return 0.6898322629693772
        if idx == 3 and coord == 'y':
            # (yvardiff3, sigmayvec3) = mapkeyfiltyvardiff(3, 0.67, 0.68, 100)
            # return findminval1d(yvardiff3**2, sigmayvec3)
            return 0.6740337546526619
        if idx == 4 and coord == 'x':
            # (xvardiff4, sigmaxvec4) = mapkeyfiltxvardiff(4, 0.75, 0.76, 100)
            # return findminval1d(xvardiff4**2, sigmaxvec4)
            return 0.756029853407984
        if idx == 4 and coord == 'y':
            # (yvardiff4, sigmayvec4) = mapkeyfiltyvardiff(4, 0.45, 0.47, 100)
            # return findminval1d(yvardiff4**2, sigmayvec4)
            return 0.4595702668204834
        if idx == 5 and coord == 'x':
            # (xvardiff5, sigmaxvec5) = mapkeyfiltxvardiff(5, 1.1, 1.11, 100)
            # return findminval1d(xvardiff5**2, sigmaxvec5)
            return 1.1066640009337496
        if idx == 5 and coord == 'y':
            # (yvardiff5, sigmayvec5) = mapkeyfiltyvardiff(5, 0.94, 0.95, 100)
            # findminval1d(yvardiff5**2, sigmayvec5)
            return 0.944777139036421
        if idx == 6 and coord == 'x':
            # (xvardiff6, sigmaxvec6) = mapkeyfiltxvardiff(6, 0.89, 0.91, 100)
            # return findminval1d(xvardiff6**2, sigmaxvec6)
            return 0.9003583091930647
        if idx == 6 and coord == 'y':
            # (yvardiff6, sigmayvec6) = mapkeyfiltyvardiff(6, 0.88, 0.90, 100)
            #  findminval1d(yvardiff6**2, sigmayvec6)
            return 0.888872322603157

        if idx == 7 and coord == 'x':
            # Use result from two-parameter l2-method
            return keyfiltersigmahat(idx, 'x', 'l2diffsamescale')
        if idx == 7 and coord == 'y':
            # Use result from two-parameter l2-method
            return keyfiltersigmahat(idx, 'y', 'l2diffsamescale')
        
        # if idx == 7 and coord == 'x':
        #     # (xvardiff7, sigmaxvec7) = mapkeyfiltxvardiff(7, 0.26, 0.28, 100)
        #     # return findminval1d(xvardiff7**2, sigmaxvec7)
        #     return 0.27203815823259314
        # if idx == 7 and coord == 'y':
        #     # (yvardiff7, sigmayvec7) = mapkeyfiltyvardiff(7, 0.1, 0.3, 100)
        #     # findminval1d(yvardiff7**2, sigmayvec7)
        #     return 0.4376643008684091

        if idx == 8 and coord == 'x':
            # (xvardiff8, sigmaxvec8) = mapkeyfiltxvardiff(8, 0.60, 0.62, 100)
            # findminval1d(xvardiff8**2, sigmaxvec8)
            return 0.6094604392295698
        if idx == 8 and coord == 'y':
            # (yvardiff8, sigmayvec8) = mapkeyfiltyvardiff(8, 0.59, 0.61, 100)
            # return findminval1d(yvardiff8**2, sigmayvec8)
            return 0.6011991729854481

        raise ValueError(f'The index ({idx}) must be between 1 and 8')

    if sigmamethod == 'jointfromdiscvar':
        if idx == 1 or idx == 2 or idx == 3 or idx == 4 or idx == 8:

            sigma1 = \
              keyfiltersigmahat(1, 'x', 'sepfromdiscvar', scspmethod) * \
              keyfiltersigmahat(1, 'y', 'sepfromdiscvar', scspmethod)
            sigma2 = \
              keyfiltersigmahat(2, 'x', 'sepfromdiscvar', scspmethod) * \
              keyfiltersigmahat(2, 'y', 'sepfromdiscvar', scspmethod)
            sigma3 = \
              keyfiltersigmahat(3, 'x', 'sepfromdiscvar', scspmethod) * \
              keyfiltersigmahat(3, 'y', 'sepfromdiscvar', scspmethod)
            sigma4 = \
              keyfiltersigmahat(4, 'x', 'sepfromdiscvar', scspmethod) * \
              keyfiltersigmahat(4, 'y', 'sepfromdiscvar', scspmethod)
            sigma8 = \
              keyfiltersigmahat(8, 'x', 'sepfromdiscvar', scspmethod) * \
              keyfiltersigmahat(8, 'y', 'sepfromdiscvar', scspmethod)

            return (sigma1 * sigma2 * sigma3 * sigma4 * sigma8) ** (1/10)

        if idx == 5 or idx == 6:
            sigma5 = \
              keyfiltersigmahat(5, 'x', 'sepfromdiscvar', scspmethod) * \
              keyfiltersigmahat(5, 'y', 'sepfromdiscvar', scspmethod)
            sigma6 = \
              keyfiltersigmahat(6, 'x', 'sepfromdiscvar', scspmethod) * \
              keyfiltersigmahat(6, 'y', 'sepfromdiscvar', scspmethod)
      
            return (sigma5 * sigma6) ** (1/4)

        if idx == 7:
            sigma7 = \
              keyfiltersigmahat(7, 'x', 'sepfromdiscvar', scspmethod) * \
              keyfiltersigmahat(7, 'y', 'sepfromdiscvar', scspmethod)

            return sigma7 ** (1/2)

        raise ValueError(f'The filter index ({idx}) must be between 1 and 8')

    if sigmamethod == 'l1diffsepscale':
        if idx == 1 and coord == 'x':
            return 0.36015075376884426
        if idx == 1 and coord == 'y':
            return 0.5104522613065327
        if idx == 2 and coord == 'x':
            return 0.5546733668341708
        if idx == 2 and coord == 'y':
            return 0.4531658291457286
        if idx == 3 and coord == 'x':
            return 0.7006532663316583
        if idx == 3 and coord == 'y':
            return 0.6551256281407035
        if idx == 4 and coord == 'x':
            return 0.562713567839196
        if idx == 4 and coord == 'y':
            return 0.3843718592964824
        if idx == 5 and coord == 'x':
            return 1.3092964824120603
        if idx == 5 and coord == 'y':
            return 0.8752763819095477
        if idx == 6 and coord == 'x':
            return 0.9728140703517588
        if idx == 6 and coord == 'y':
            return 1.1711557788944722    
        if idx == 7 and coord == 'x':
            # Use result from two-parameter l2-method
            return keyfiltersigmahat(idx, 'x', 'l2diffsamescale')
        if idx == 7 and coord == 'y':
            # Use result from two-parameter l2-method
            return keyfiltersigmahat(idx, 'y', 'l2diffsamescale')
        if idx == 8 and coord == 'x':
            return 0.6366331658291458
        if idx == 8 and coord == 'y':
            return 0.5866331658291457
        
        raise ValueError(f'The index ({idx}) must be between 1 and 8')

    
    if sigmamethod == 'l1diffsamescale':
        if idx == 1:
            return 0.4580194313955851
        if idx == 2:
            return 0.4482463177756579
        if idx == 3:
            return 0.6711849867963244
        if idx == 4:
            return 0.4203522130670464
        if idx == 5:
            return 1.3873955463956178
        if idx == 6:
            return 1.0902606031644924
        if idx == 7:
            # From two-parameter optimization with alpha
            return 0.6537688442211055
        if idx == 8:
            return 0.6119728613663912
                
        raise ValueError(f'The index ({idx}) must be between 1 and 8')
    
    if sigmamethod == 'l1diffjointscale':
        if idx == 1 or idx == 2 or idx == 3 or idx == 4 or idx == 8:
            return 0.6074435027601324
        if idx == 5 or idx == 6:
            return 1.1329249680353086
        if idx == 7:
            # From two-parameter optimization with alpha
            return 0.6537688442211055
                
        raise ValueError(f'The index ({idx}) must be between 1 and 8')

    if sigmamethod == 'l2diffsepscale':
        if idx == 1 and coord == 'x':
            return 0.4912060301507538
        if idx == 1 and coord == 'y':
            return 0.7224120603015075
        if idx == 2 and coord == 'x':
            return 0.581356783919598
        if idx == 2 and coord == 'y':
            return 0.518643216080402
        if idx == 3 and coord == 'x':
            return 0.48311557788944726
        if idx == 3 and coord == 'y':
            return 0.5028643216080402
        if idx == 4 and coord == 'x':
            return 0.5009547738693467
        if idx == 4 and coord == 'y':
            return 0.0002
        if idx == 5 and coord == 'x':
            return 1.299748743718593
        if idx == 5 and coord == 'y':
            return 1.0042713567839197
        if idx == 6 and coord == 'x':
            return 0.9842713567839196
        if idx == 6 and coord == 'y':
            return 1.0737688442211055
        if idx == 7 and coord == 'x':
            # Use result from two-parameter l2-method
            return keyfiltersigmahat(idx, 'x', 'l2diffsamescale')
        if idx == 7 and coord == 'y':
            # Use result from two-parameter l2-method
            return keyfiltersigmahat(idx, 'y', 'l2diffsamescale')
        if idx == 8 and coord == 'x':
            return 0.6156281407035176
        if idx == 8 and coord == 'y':
            return 0.608140703517588
    
        raise ValueError(f'The index ({idx}) must be between 1 and 8')
            
    if sigmamethod == 'l2diffsamescale':
        if idx == 1:
            return 0.6440209979383456
        if idx == 2:
            return 0.5580021518662255
        if idx == 3:
            return 0.4951507717698759
        if idx == 4:
            return 0.37962065829063185
        if idx == 5:
            return 1.1934352970554198
        if idx == 6:
            return 1.0382624689911877
        if idx == 7:
            # From two-parameter optimization with alpha
            return 0.6753768844221105
        if idx == 8:
            return 0.6118535135425845
                
        raise ValueError(f'The index ({idx}) must be between 1 and 8')

    if sigmamethod == 'l2diffjointscale':
        if idx == 1 or idx == 2 or idx == 3 or idx == 4 or idx == 8:
            return 0.5216205270928954
        if idx == 5 or idx == 6:
            return 1.10817570616526
        if idx == 7:
            # From two-parameter optimization with alpha
            return 0.6753768844221105
                
        raise ValueError(f'The index ({idx}) must be between 1 and 8')
    
    raise ValueError(f'Unknown sigmamethod ({sigmamethod})')


def keyfilter7alphahat(
        sigmamethod : 'str'
    ) -> float :
    """Returns the estimate of the factor alpha in the idealizel mode
    of key filter 7
    """
    if sigmamethod == 'l1diffsamescale':
        return 0.521608040201005
    if sigmamethod == 'l2diffsamescale':
        return 0.5256281407035176

    raise ValueError(f'Unknown sigmamethod ({sigmamethod})')

    
def keyfiltertemplweightvariance(
        idx : int,
        sigmax : float,
        sigmay : float,
        sigmaxref : float,
        sigmayref : float,
        scspmethod : str = 'discgauss',
        epsilon : float = 0.00000001
    ) -> np.array :
    """Computes the weighted variance-based of an idealized template filter of a 
    structurally similar shape as the 'master key filter' with index idx, 
    given values of the spatial scale parameters sigmax and sigmay for
    the ideal filter, and the spatial scale parameter sigmaxref and sigmayref
    for the weighting filter, in the horizontal and vertical directions, respectively."""
    
    idealfilter = keyfiltertempl(idx, sigmax, sigmay, scspmethod, epsilon)
    weightfilter = keyfiltertempl(idx, sigmaxref, sigmayref, scspmethod, epsilon)

    return weightvariance(np.abs(idealfilter), np.abs(weightfilter))


def sigmaarray(
    sigmamin : float,
    sigmamax : float,
    numsamples : int
    ) -> np.array :
    """Creates an array of logarithmically distributed scale values
    """
    logscales = np.linspace(log(sigmamin), log(sigmamax), numsamples)

    return np.exp(logscales)


def mapkeyfilttemplweightxvardiff(
        idx : int,
        sigmaxmin : float,
        sigmaxmax : float,
        numsamples : int,
        sigmay : float,
        sigmaxref : float,
        sigmayref : float,
        scspmethod : str = 'discgauss',
        epsilon : float = 0.00000001
    ) -> (np.array, np.array) :
    """Computes the difference between the horizontal x components of the
    weighted variance-based spatial spread measures computed from an ideal
    filter with idx and the corresponding normalized learned filter with
    the same index idx. This difference is computed for a logaritmically
    distributed horizontal scale values in the interval [sigmaxmin, sigmaxmax]
    and with the horizontal and vertical scale parameters sigmaxref and
    sigmayref for the ideal spatial weighting function."""
    
    sigmavec = sigmaarray(sigmaxmin, sigmaxmax, numsamples)
    out = np.zeros(len(sigmavec))

    # Compared to the variance
    learnedkeyfiltvar = \
      keyfiltweightvariance(idx, sigmaxref, sigmayref, scspmethod, epsilon)

    for i, sigmax in np.ndenumerate(sigmavec):
        idealkeyfiltvar = \
          keyfiltertemplweightvariance(idx, sigmax, sigmay, \
                                       sigmaxref, sigmayref, scspmethod, epsilon)

        out[i] = idealkeyfiltvar[0][0] - learnedkeyfiltvar[0][0]

    return out, sigmavec


def mapkeyfilttemplweightyvardiff(
        idx : int,
        sigmax : float,
        sigmaymin : float,
        sigmaymax : float,
        numsamples : int,
        sigmaxref : float,
        sigmayref : float,
        scspmethod : str = 'discgauss',
        epsilon : float = 0.00000001
    ) -> (np.array, np.array) :
    """Computes the difference between the vertical y components of the
    weighted variance-based spatial spread measures computed from an ideal
    filter with idx and the corresponding normalized learned filter with
    the same index idx. This difference is computed for a logaritmically
    distributed vertical scale values in the interval [sigmaymin, sigmaymax]
    and with the horizontal and vertical scale parameters sigmaxref and
    sigmayref for the ideal spatial weighting function."""
    
    sigmavec = sigmaarray(sigmaymin, sigmaymax, numsamples)
    out = np.zeros(len(sigmavec))

    # Compared to the variance
    learnedkeyfiltvar = \
      keyfiltweightvariance(idx, sigmaxref, sigmayref, scspmethod, epsilon)

    for i, sigmay in np.ndenumerate(sigmavec):
        idealkeyfiltvar = \
          keyfiltertemplweightvariance(idx, sigmax, sigmay, \
                                       sigmaxref, sigmayref, scspmethod, epsilon)

        out[i] = idealkeyfiltvar[1][1] - learnedkeyfiltvar[1][1]

    return out, sigmavec


def mapkeyfiltxvardiff(
        idx : int,
        sigmaxmin : float,
        sigmaxmax : float,
        numsamples : int,
        scspmethod : str = 'discgauss',
        epsilon : float = 0.00000001
    ) -> (np.array, np.array) :
    """Syntactic sugar with default parameters for the function mapkeyfilttemplweightxvardiff
    """
    if idx == 7:
        sigmay = 1/sqrt(2)
        sigmaxref = 1/sqrt(2)
        sigmayref = 1/sqrt(2)
    else:
        sigmay = 1.0
        sigmaxref = 1.0
        sigmayref = 1.0

    return mapkeyfilttemplweightxvardiff(idx, sigmaxmin, sigmaxmax, numsamples, \
                                         sigmay, sigmaxref, sigmayref, scspmethod, epsilon)


def mapkeyfiltyvardiff(
        idx : int,
        sigmaymin : float,
        sigmaymax : float,
        numsamples : int,
        scspmethod : str = 'discgauss',
        epsilon : float = 0.00000001
    ) -> (np.array, np.array) :
    """Syntactic sugar with default parameters for the function mapkeyfilttemplweightyvardiff
    """
    if idx == 7:
        sigmax = 0.707
        sigmaxref = 0.707
        sigmayref = 0.707
    else:
        sigmax = 1.0
        sigmaxref = 1.0
        sigmayref = 1.0

    return mapkeyfilttemplweightyvardiff(idx, sigmax, sigmaymin, sigmaymax, numsamples, \
                                         sigmaxref, sigmayref, scspmethod, epsilon)


def keyfilterdiff(
        idx : int,
        sigmax : float,
        sigmay : float,
        scspmethod : str = 'discgauss',
        epsilon : float = 0.00000001
    ) -> np.array :
    """Returns the difference between the idealized template filter and the
    corresponding normalized filter for the 'master key filter" with index idx.
    """
    templfilter = keyfiltertempl(idx, sigmax, sigmay, scspmethod, epsilon)
    normfilter = normkeyfilter(idx)

    return templfilter - normfilter


def keyfilterdiff7(
        sigma : float,
        alpha : float,
        scspmethod : str = 'discgauss',
        epsilon : float = 0.00000001
    ) -> np.array :
    """Returns the difference between the idealized template filter and the
    corresponding normalized filter for the 'master key filter" with index 7.
    """
    templfilter7 = keyfiltertempl7(sigma, alpha, scspmethod, epsilon)
    normfilter7 = normkeyfilter(7)

    return templfilter7 - normfilter7


def normkeyfilter(idx : int) -> np.ndarray :
    """Returns the normalized version of the 'master key filter' with index idx.
    """
    if idx == 1:
        return keyfilter('norm1')
    if idx == 2:
        return keyfilter('norm2')
    if idx == 3:
        return keyfilter('norm3')
    if idx == 4:
        return keyfilter('norm4')
    if idx == 5:
        return keyfilter('norm5')
    if idx == 6:
        return keyfilter('norm6')
    if idx == 7:
        return keyfilter('norm7')
    if idx == 8:
        return keyfilter('norm8')

    raise ValueError(f'The filter index ({idx}) must be between 1 and 8')


def mapkeyfilterl1diffsamescale(
        idx : int,
        sigmamin : float,
        sigmamax : float,
        numsamples : int,
        scspmethod : str = 'discgauss',
        epsilon : float = 0.00000001
    ) -> np.array :
    """Computes the l1-norm of the difference between the idealized version of
    the 'master key filter' with index idx and the normalized version of the
    corresponding learned filter. This measure is computed for a logaritmically
    distributed vertical scale values in the interval [sigmamin, sigmamax].
    """
    sigmavec = sigmaarray(sigmamin, sigmamax, numsamples)
    out = np.zeros(len(sigmavec))

    for i, sigma in np.ndenumerate(sigmavec):
        out[i] = L1norm(keyfilterdiff(idx, sigma, sigma, scspmethod, epsilon))

    return out, sigmavec

   
def mapkeyfilterl2diffsamescale(
        idx : int,
        sigmamin : float,
        sigmamax : float,
        numsamples : int,
        scspmethod : str = 'discgauss',
        epsilon : float = 0.00000001
    ) -> np.array :
    """Computes the l2-norm of the difference between the idealized version of
    the 'master key filter' with index idx and the normalized version of the
    the corresponding learned filter. This measure is computed for a logaritmically
    distributed vertical scale values in the interval [sigmamin, sigmamax].
    """
    sigmavec = sigmaarray(sigmamin, sigmamax, numsamples)
    out = np.zeros(len(sigmavec))

    for i, sigma in np.ndenumerate(sigmavec):
        out[i] = L2norm(keyfilterdiff(idx, sigma, sigma, scspmethod, epsilon))

    return out, sigmavec


def L2norm(
        discfilter : np.ndarray
    ) -> float :
    """Computes the l2-norm of a discrete filter"""

    return (np.sum(discfilter**2))**(1/2)


def mapkeyfilterl1diffjoint12348(
        sigmamin : float,
        sigmamax : float,
        numsamples : int,
        scspmethod : str = 'discgauss',
        epsilon : float = 0.00000001
    ) -> np.array :
    """Computes the pooled l1-norm over the group of filters {1, 2, 3, 4, 8}
    for the difference between the idealized version of the 'master key filter' 
    with index idx and the normalized version of the corresponding learned filter. 
    This measure is computed for a logaritmically distributed vertical scale 
    values in the interval [sigmamin, sigmamax].
    """
    part1, scales = \
      mapkeyfilterl1diffsamescale(1, sigmamin, sigmamax, numsamples, scspmethod, epsilon)
    part2, scales = \
      mapkeyfilterl1diffsamescale(2, sigmamin, sigmamax, numsamples, scspmethod, epsilon)
    part3, scales = \
      mapkeyfilterl1diffsamescale(3, sigmamin, sigmamax, numsamples, scspmethod, epsilon)
    part4, scales = \
      mapkeyfilterl1diffsamescale(4, sigmamin, sigmamax, numsamples, scspmethod, epsilon)
    part8, scales = \
      mapkeyfilterl1diffsamescale(8, sigmamin, sigmamax, numsamples, scspmethod, epsilon)

    return part1 + part2 + part3 + part4 + part8, scales


def mapkeyfilterl1diffjoint56(
        sigmamin : float,
        sigmamax : float,
        numsamples : int,
        scspmethod : str = 'discgauss',
        epsilon : float = 0.00000001
    ) -> np.array :
    """Computes the pooled l1-norm over the group of filters {5, 6}
    for the difference between the idealized version of the 'master key filter' 
    with index idx and the normalized version of the corresponding learned filter. 
    This measure is computed for a logaritmically distributed vertical scale 
    values in the interval [sigmamin, sigmamax].
    """
    part5, scales = \
      mapkeyfilterl1diffsamescale(5, sigmamin, sigmamax, numsamples, scspmethod, epsilon)
    part6, scales = \
      mapkeyfilterl1diffsamescale(6, sigmamin, sigmamax, numsamples, scspmethod, epsilon)

    return part5 + part6, scales


def mapkeyfilterl2diffjoint12348(
        sigmamin : float,
        sigmamax : float,
        numsamples : int,
        scspmethod : str = 'discgauss',
        epsilon : float = 0.00000001
    ) -> np.array :
    """Computes the pooled l2-norm over the group of filters {1, 2, 3, 4, 8}
    for the difference between the idealized version of the 'master key filter' 
    with index idx and the normalized version of the corresponding learned filter. 
    This measure is computed for a logaritmically distributed vertical scale 
    values in the interval [sigmamin, sigmamax].
    """
    part1, scales = \
      mapkeyfilterl2diffsamescale(1, sigmamin, sigmamax, numsamples, scspmethod, epsilon)
    part2, scales = \
      mapkeyfilterl2diffsamescale(2, sigmamin, sigmamax, numsamples, scspmethod, epsilon)
    part3, scales = \
      mapkeyfilterl2diffsamescale(3, sigmamin, sigmamax, numsamples, scspmethod, epsilon)
    part4, scales = \
      mapkeyfilterl2diffsamescale(4, sigmamin, sigmamax, numsamples, scspmethod, epsilon)
    part8, scales = \
      mapkeyfilterl2diffsamescale(8, sigmamin, sigmamax, numsamples, scspmethod, epsilon)

    return part1**2 + part2**2 + part3**2 + part4**2 + part8**2, scales


def mapkeyfilterl2diffjoint56(
        sigmamin : float,
        sigmamax : float,
        numsamples : int,
        scspmethod : str = 'discgauss',
        epsilon : float = 0.00000001
    ) -> np.array :
    """Computes the pooled l2-norm over the group of filters {5, 6}
    for the difference between the idealized version of the 'master key filter' 
    with index idx and the normalized version of the corresponding learned filter. 
    This measure is computed for a logaritmically distributed vertical scale 
    values in the interval [sigmamin, sigmamax].
    """
    part5, scales = \
      mapkeyfilterl2diffsamescale(5, sigmamin, sigmamax, numsamples, scspmethod, epsilon)
    part6, scales = \
      mapkeyfilterl2diffsamescale(6, sigmamin, sigmamax, numsamples, scspmethod, epsilon)

    return part5**2 + part6**2, scales


def mapkeyfilter7l1diff(
        sigmamin : float,
        sigmamax : float,
        alphamin : float,
        alphamax : float,
        numsamples : int,
        scspmethod : str = 'discgauss',
        epsilon : float = 0.00000001
    ) -> np.array :
    """Computes the l1-norm of the difference between the idealized version of
    the 'master key filter' with index idx and the normalized version of the
    corresponding learned filter. This measure is computed for a logaritmically
    distributed vertical scale values in the interval [sigmamin, sigmamax].
    """
    sigmavec = np.linspace(sigmamin, sigmamax, numsamples)
    alphavec = np.linspace(alphamin, alphamax, numsamples)
    out = np.zeros((len(sigmavec), len(alphavec)))

    for i, sigma in np.ndenumerate(sigmavec):
        for j, alpha in np.ndenumerate(alphavec):
            out[i][j] = L1norm(keyfilterdiff7(sigma, alpha, scspmethod, epsilon))

    return out, sigmavec, alphavec



def mapkeyfilter7l2diff(
        sigmamin : float,
        sigmamax : float,
        alphamin : float,
        alphamax : float,
        numsamples : int,
        scspmethod : str = 'discgauss',
        epsilon : float = 0.00000001
    ) -> np.array :
    """Computes the l2-norm of the difference between the idealized version of
    the 'master key filter' with index idx and the normalized version of the
    corresponding learned filter. This measure is computed for a logaritmically
    distributed vertical scale values in the interval [sigmamin, sigmamax].
    """
    sigmavec = np.linspace(sigmamin, sigmamax, numsamples)
    alphavec = np.linspace(alphamin, alphamax, numsamples)
    out = np.zeros((len(sigmavec), len(alphavec)))

    for i, sigma in np.ndenumerate(sigmavec):
        for j, alpha in np.ndenumerate(alphavec):
            out[i][j] = L2norm(keyfilterdiff7(sigma, alpha, scspmethod, epsilon))

    return out, sigmavec, alphavec


def mapkeyfiltertwoscalel1diff(
        idx : int,
        sigmaxmin : float,
        sigmaxmax : float,
        sigmaymin : float,
        sigmaymax : float,
        numsamples : int,
        scspmethod : str = 'discgauss',
        epsilon : float = 0.00000001
    ) -> np.array :
    """Computes the l2-norm of the difference between the idealized version of
    the 'master key filter' with index idx and the normalized version of the
    corresponding learned filter, under variations of both the horizontal
    and the vertical scale parameters. This measure is computed for a 
    logaritmically distributed vertical scale values in the interval 
    [sigmamin, sigmamax].
    """
    sigmaxvec = np.linspace(sigmaxmin, sigmaxmax, numsamples)
    sigmayvec = np.linspace(sigmaymin, sigmaymax, numsamples)
    out = np.zeros((len(sigmaxvec), len(sigmayvec)))

    for i, sigmax in np.ndenumerate(sigmaxvec):
        for j, sigmay in np.ndenumerate(sigmayvec):
            out[i][j] = \
              L1norm(keyfilterdiff(idx, sigmax, sigmay, scspmethod, epsilon))

    return out, sigmaxvec, sigmayvec


def mapkeyfiltertwoscalel2diff(
        idx : int,
        sigmaxmin : float,
        sigmaxmax : float,
        sigmaymin : float,
        sigmaymax : float,
        numsamples : int,
        scspmethod : str = 'discgauss',
        epsilon : float = 0.00000001
    ) -> np.array :
    """Computes the l2-norm of the difference between the idealized version of
    the 'master key filter' with index idx and the normalized version of the
    corresponding learned filter, under variations of both the horizontal
    and the vertical scale parameters. This measure is computed for a 
    logaritmically distributed vertical scale values in the interval 
    [sigmamin, sigmamax].
    """
    sigmaxvec = np.linspace(sigmaxmin, sigmaxmax, numsamples)
    sigmayvec = np.linspace(sigmaymin, sigmaymax, numsamples)
    out = np.zeros((len(sigmaxvec), len(sigmayvec)))

    for i, sigmax in np.ndenumerate(sigmaxvec):
        for j, sigmay in np.ndenumerate(sigmayvec):
            out[i][j] = \
              L2norm(keyfilterdiff(idx, sigmax, sigmay, scspmethod, epsilon))

    return out, sigmaxvec, sigmayvec
