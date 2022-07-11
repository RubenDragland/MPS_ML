
import numpy as np
import scipy
import matplotlib.pyplot as plt
from numba import njit
#from numpy.linalg import eigvalsh, norm, svd
from scipy.linalg import svd

### Not necessary in this file
import torch.nn as nn
import pytorch_lightning as pl


class MPS:
    """Class for a matrix product state.

    We index sites with `i` from 0 to L-1; bond `i` is left of site `i`.
    We *assume* that the state is in right-canonical form.

    Parameters
    ----------
    Bs, Ss:
        Same as attributes.

    Attributes
    ----------
    Bs : list of np.Array[ndim=3]
        The 'matrices' in right-canonical form, one for each physical site.
        Each `B[i]` has legs (virtual left, physical, virtual right), in short ``vL i vR``
    Ss : list of np.Array[ndim=1]
        The Schmidt values at each of the bonds, ``Ss[i]`` is left of ``Bs[i]``.
    L : int
        Number of sites.
    """

    def __init__(self, Bs, Ss):
        self.Bs = Bs
        self.Ss = Ss
        self.L = len(Bs)

    def copy(self):
        return MPS([B.copy() for B in self.Bs], [S.copy() for S in self.Ss])

    def get_theta1(self, i):
        """Calculate effective single-site wave function on sites i in mixed canonical form.

        The returned array has legs ``vL, i, vR`` (as one of the Bs)."""
        return np.tensordot(np.diag(self.Ss[i]), self.Bs[i], [1, 0])  # vL [vL'], [vL] i vR

    def get_theta2(self, i):
        """Calculate effective two-site wave function on sites i,j=(i+1) in mixed canonical form.

        The returned array has legs ``vL, i, j, vR``."""
        j = i + 1
        return np.tensordot(self.get_theta1(i), self.Bs[j], [2, 0])  # vL i [vR], [vL] j vR

    def get_chi(self):
        """Return bond dimensions."""
        return [self.Bs[i].shape[2] for i in range(self.L - 1)]

    def site_expectation_value(self, op):
        """Calculate expectation values of a local operator at each site."""
        result = []
        for i in range(self.L):
            theta = self.get_theta1(i)  # vL i vR
            op_theta = np.tensordot(op, theta, axes=[1, 1])  # i [i*], vL [i] vR
            result.append(np.tensordot(theta.conj(), op_theta, [[0, 1, 2], [1, 0, 2]]))
            # [vL*] [i*] [vR*], [i] [vL] [vR]
        return np.real_if_close(result)

    def bond_expectation_value(self, op):
        """Calculate expectation values of a local operator at each bond."""
        result = []
        for i in range(self.L - 1):
            theta = self.get_theta2(i)  # vL i j vR
            op_theta = np.tensordot(op[i], theta, axes=[[2, 3], [1, 2]])
            # i j [i*] [j*], vL [i] [j] vR
            result.append(np.tensordot(theta.conj(), op_theta, [[0, 1, 2, 3], [2, 0, 1, 3]]))
            # [vL*] [i*] [j*] [vR*], [i] [j] [vL] [vR]
        return np.real_if_close(result)

    def entanglement_entropy(self):
        """Return the (von-Neumann) entanglement entropy for a bipartition at any of the bonds."""
        result = []
        for i in range(1, self.L):
            S = self.Ss[i].copy()
            S[S < 1.e-20] = 0.  # 0*log(0) should give 0; avoid warning or NaN.
            S2 = S * S
            assert abs(np.linalg.norm(S) - 1.) < 1.e-14
            result.append(-np.sum(S2 * np.log(S2)))
        return np.array(result)


class MNIST_MPS(MPS):

    def __init__(self, img: np.ndarray, spin_dimensions: tuple, label: int, periodic = True ):
        # Save the input
        self.dim = spin_dimensions
        self.L = self.dim[0]*self.dim[1]
        self.img = img # Might be shit to store...
        self.image_1d = self.flatten_rescale_pixels(img)
        self.label = label

        self.PBC = periodic

        Bs, Ss = self.quantum_states()
        super().__init__( Bs, Ss) # Think this works. 

    def __add__(self, other):

        if self.PBC:

            assert other.PBC
            assert self.label == other.label

            # Assume periodic boundaries are eq. to open. In addition, we assume we can work with gauge degrees of freedom
            # Tr(A+B) -> Tr(A) + Tr(B). Thus
            Bs = []
            Ss = []
            for i in range( self.L ):
                Ss.append( (self.Ss[i] + other.Ss[i])/ np.linalg.norm(self.Ss[i] + other.Ss[i]) )
                Bs.append( (self.Bs[i] + other.Bs[i])/ np.linalg.norm(self.Bs[i][0,:,0] + other.Bs[i][0,:,0]) )
                # Normalize each spin. 
            
            xj_from_Bs = lambda  Bs : np.arccos(Bs)*2*np.pi

            B_values = np.array(Bs)[:,0,:,0]
            xj = xj_from_Bs(B_values)[:,0]
            new_img = np.reshape(xj, self.dim)

            sum_MPS = MNIST_MPS(img= new_img, spin_dimensions= self.dim, label = self.label) 
            sum_MPS.Bs = Bs
            sum_MPS.Ss = Ss # Have not normalized. Assertion would be better?
            sum_MPS.normalize_wavefunction() # Normalize new WF at each pixel. (Assertion not necessary)

            # A lot of unefficient shit; Needs to be done in a better way. If the above does not work:
            #B_values = np.array(sum_MPS.Bs)[:,0,:,0]
            #xj = xj_from_Bs(B_values)
            #new_img = np.reshape(xj, self.dim)
            #sum_MPS.img = new_img
            #sum_MPS.image_1d = sum_MPS.flatten_rescale_pixels(sum_MPS.img)
            
            return sum_MPS

        
        else:
  
            new_img = self.img + other.img # Nope do inverse transform

            row_vector_sig1 = np.array( [ self.get_theta1(0), other.get_theta1(0) ] ) #np.array( [self.Bs[0], other.Bs[0] ] ) # nope need mixed canonical
            col_vector_sigL = np.array( [ self.get_theta1(self.L-1), other.get_theta1(self.L-1) ] ) #np.array( [self.Bs[-1], other.Bs[-1] ] ) #Transpose?

            #sum_MPS = MNIST_MPS(new_img, self.dim) # Place holder for sum of two MNIST_MPS Fill it up, and correct the image.
            Ns = [row_vector_sig1]

            #May actually be possible to do this without merging to thetas? But easier to follow recipe.
            for i in range(1, self.L-1 ):
                M = self.get_theta1(i)
                M_th = other.get_theta1()   # Nope, think actually that we are to use get_theta2, or not actually. Only 1 physical state

                N = np.zeros(np.add(M.shape, M_th.shape ) )
                print(N.shape)
                N[:M.shape[0], :M.shape[1] ] = M
                N[M.shape[0]:, M.shape[1]: ] = M_th
                Ns.append(N)
            
            Ns.append(col_vector_sigL)

            # Here, some compression should be done. 

            # End compression

            for j in range(self.L-1, -1, -1):

                # Remember correct shape of theta. Need to reshape with sig and another index together. Retrieve one index and use the others further. 
                X,Y,Z = svd(Ns[j][0,:,0], lapack_driver = 'gesvd') # Check shapes
                #Truncation, since we already do a svd?

                # Define new Bs and Ss

                # Continue if necessary



    # Do everything, and make inverse transform to create the image. In this way, one may return an image. 
        return 
    
    def __radd__(self, other):

        return other.__add__(self)
    
    def __len__(self):
        return self.L


    def flatten_rescale_pixels(self, img):
        img = img.reshape( self.L )
        return (img - np.min(img) ) / ( np.max(img) - np.min(img) ) 
    
    def quantum_states(self):
            qs = np.zeros( ( self.L, 2 ) )
            phi_j = lambda xj: np.array( [np.cos(np.pi/2 * xj), np.sin(np.pi/2 *xj )] ) # Check shape
            qs[:,:] = phi_j( self.image_1d ).T # Must be transposed?

            B = np.zeros([len(qs), 1, 2, 1], np.float)
            B[:,0, :, 0] = qs[:,:] # Check shape

            S = np.ones([1], np.float)
            Bs = [B[i].copy() for i in range(self.L)]
            Ss = [S.copy() for i in range(self.L)]
            return Bs, Ss
    
    def normalize_wavefunction(self):
        #norm_2 =  overlap_rC(self, self) # Might Not necessary; I think. Do testign

        self.Bs = [self.Bs[i].copy() / np.linalg.norm(self.Bs[i][0,:,0]) for i in range(self.L)]
        self.Ss = [self.Ss[i].copy() / np.linalg.norm(self.Ss[i]) for i in range(self.L)]

        return # Not sure if correct, but probably

    
    ## From exercise 7, not sure if necessary. Known schmidt values equal to 1 since pure product state. 
    #def compress_to_mps(self, chimax) -> list:
    #    def quantum_states(self):
    #        qs = np.zeros( ( self.L, 2 ) )
    #        phi_j = lambda xj: np.array( np.cos(np.pi/2 * xj), np.sin(np.pi/2 *xj ) )
    #        qs[:,:] = phi_j( self.image_1d ).T # Must be transposed
    #        return qs
    #
    #    dim_R = lambda n, l : 2**(l-(n-1)) 
    #    L = self.L
    #
    #    psi = self.quantum_states() # Get expression of each pixel as a qm state. Like init_spinup or spin_right
    #
    #    compressed = []
    #    psi_aR = np.reshape(psi, (1, 2**L ))
    #
    #    for n in range(1,L+1):
    #        chi_n = psi_aR.shape[0]
    #        psi_alfa_R = np.reshape(psi_aR, (2*chi_n, dim_R(n, L) // 2 ) )
    #
    #        M_n, lambda_n, psitilde = svd(psi_alfa_R, full_matrices = False, lapack_driver='gesvd' )
    #        
    #        if len(lambda_n) > chimax: # Include some eps here like split and truncate.
    #            keep = np . argsort ( lambda_n )[:: -1][: chimax ]
    #            M_n = M_n [: , keep ]
    #            lambda_n = lambda_n [ keep ]
    #            psitilde = psitilde [ keep , :]
    #        
    #        chi_n1 = len(lambda_n) # Make Ss or something
    #        
    #        M_n = np.reshape(M_n, (chi_n, 2, chi_n1 ) )  # Truncation necessary
    #        compressed.append(M_n) # Make Bs here
    #        psi_aR = lambda_n[:, np.newaxis] * psitilde[:, :] 
    #        
    #    return compressed

    
        # Convert to spins
        # Convert to MPS and compress
        # Overload __add__

        # See ex 7 amongst others

        # Note think mixed canonical. Use split and truncate, or do something else self-defined and clever. 
        # Or is it possible to create MPS without this compression? Check equations
        # Include eps so not chimax only defining factor. See truncation
        # https://en.wikipedia.org/wiki/Schmidt_decomposition https://physics.stackexchange.com/questions/251522/how-do-you-find-a-schmidt-basis-and-how-can-the-schmidt-decomposition-be-used-f 
        # https://www.rle.mit.edu/cua_pub/8.422/Reading%20Material/NOTES-schmidt-decomposition-and-epr-from-nielsen-and-chuang-p109.pdf
        # I.e. each image is a product state and has zero entanglement -> Schmidt values are 1. 


        # Thoughts 20220710
        # Normalize both Ss and Bs. Ss is nevertheless supposed to be 1s all over. 
        # Normalization by ensuring that each pixel  
        # Quite inefficient addition right now.


        # Answers and Thoughts 20220711
        # PBC and canonical form does not work. Why?
        # The direct sum increases dimensionality for OBC and PBC. But What if one applies eq134 in XXX? This was done in the first draft, and seemed to work.
        # However, there is really no need for canonical form, as the best working overlap function does not use canonical form. 
        # Both overlap functions now work. 
        # Best thing to do is to finish OBC and compare. If drastically different, discuss.
        # Also, finish the rescaling of pixels and the NN. The addition is a separated phenomena. 

def overlap(bra, ket):
    L = len(bra)
    contr = np.ones((1,1))
    for n in range(L):
        
        M_ket = ket[n]
        contr = np.tensordot(contr, M_ket, axes = (1,0) )
        
        M_bra = bra[n].conj()
        contr = np.tensordot(M_bra, contr, axes = ([0,1], [0,1]) )
   
    return contr

def overlap_rC(bra:MNIST_MPS, ket:MNIST_MPS):
    L = len(bra)
    contr = np.ones((1,1)) # has indices (alpha_n*, alpha_n)
    #print(contr, contr.shape)

    for n in range(L-1):

        S_ket = np.diag(ket.Ss[n])      # has indices (alpha_n, alpha_{n+1}) 
        B_ket = ket.Bs[n]    # has indices (alpha_{n+1}, j_{n}, alpha_{n+2})
        contr = np.tensordot(contr, S_ket, axes = (1,0) )
        #print(contr, contr.shape)
        # now contr has indices alpha_n*, alpha_{n+1}
        #print(contr.shape)
        contr = np.tensordot(contr, B_ket, axes = (1,0) )
        # now contr has indices alpha_n*, j_n, alpha_{n+2}
        
        S_bra = np.diag(bra.Ss[n].conj() ) # has indices (alpha_n*, alpha_{n+1}*)
        B_bra = bra.Bs[n].conj() # has indices (alpha_{n+1}*, j_{n}, alpha_{n+2}*)
        contr = np.tensordot(contr, S_bra, axes = (0, 0) )
        # now contr has indices  j_n, alpha_{n+2}, alpha_{n+1}*       
        contr = np.tensordot(contr, B_bra, axes = ([2,0], [0,1]) )
        # now contr has indices 
        
    return contr.item()

def overlap_theta(bra: MNIST_MPS, ket:MNIST_MPS):
    L = len(bra)
    contr = np.ones((1,1))

    for n in range(L):
        
        M_ket = ket.get_theta1(n)
        contr = np.tensordot(contr, M_ket, axes = (1,0) )
        
        M_bra = bra.get_theta1(n).conj()
        contr = np.tensordot(M_bra, contr, axes = ([0,1], [0,1]) )
   
    return contr.item()

