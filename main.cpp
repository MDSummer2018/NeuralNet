#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <chrono>
#include <random>
#include <thread>
#include <mutex>
#include <random>
#include <limits>

#include "seal/seal.h"

using namespace std;
using namespace seal;

/*
Helper function: Prints the parameters in a SEALContext.
*/
void print_parameters(const SEALContext &context)
{
    cout << "/ Encryption parameters:" << endl;
    cout << "| poly_modulus: " << context.poly_modulus().to_string() << endl;

    /*
    Print the size of the true (product) coefficient modulus
    */
    cout << "| coeff_modulus size: " 
        << context.total_coeff_modulus().significant_bit_count() << " bits" << endl;

    cout << "| plain_modulus: " << context.plain_modulus().value() << endl;
    cout << "\\ noise_standard_deviation: " << context.noise_standard_deviation() << endl;
    cout << endl;
}

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <math.h>

#include "nn.dat"

#define NUM_LAYERS 6

float activation(int p, float z){

  float ret;
  
  if (p == 0){ // sigmoid
        ret=1/(1+exp(-z));
  } else {
    if (p == 1){ // SoftPlus
      ret=log(1+exp(z));
    } else {
      if (p == 2){ // identity
        ret=z;
      } else {
	if (p == 3){ //squaring
          ret=z*z;
	} else {
          if (p == 7){ // squaring scaled for [-8, 8]
            ret = (z/8.0)*(z/8.0);
          } else {
            if (p == 8){ // squaring scaled for [-300, 300]
              ret = (z/300.0)*(z/300.0);
            }
          }
	}
      }
    }
  }
  return ret;
}

void run_testing_net(int p[NUM_LAYERS + 1],
	             float a0[28][28],
	             float w1[5][5][5],
	             float w3[10][5][5],
	             float b5[100], float w5[100][10][5][5][5],
	             float b6[10], float w6[10][100],
	             float a6[10]){

  float z1[5][13][13];
  float a1[5][13][13];
  float a2[5][13][13];
  float z3[10][5][5][5];
  float a3[10][5][5][5];
  float a4[10][5][5][5];
  float z5[100];
  float a5[100];
  float z6[10];
  
  int j, k, m, n, r, c, a, b;

  // apply layer 1
  for (m=0; m<5; m++){
    for (c=0; c<13; c++){
      for (r=0; r<13; r++){
        z1[m][r][c] = 0;
	for (b=0; b<5; b++){
	  for (a=0; a<5; a++){
	    if ((2*r + a < 28) && (2*c + b < 28)){
              z1[m][r][c]+= w1[m][a][b]*a0[2*r + a][2*c + b];
	    }
	  }
	}
        a1[m][r][c]=activation(p[1], z1[m][r][c]);
      }
    }
  }

  // apply layer 2
  for (m=0; m<5; m++){
    for (c=0; c<13; c++){
      for (r=0; r<13; r++){
	a2[m][r][c]=0;
        for (b=0; b<3; b++){
	  for (a=0; a<3; a++){
	    if ((r + a < 13) && (c + b < 13)){
              a2[m][r][c]+=a1[m][r + a][c + b]/(float)9;
	    }
	  }
	}
      }
    }
  }

  // apply layer 3
  for (n=0; n<10; n++){
    for (m=0; m<5; m++){
      for (c=0; c<5; c++){
        for (r=0; r<5; r++){
          z3[n][m][r][c]=0;
	  for (b=0; b<5; b++){
	    for (a=0; a<5; a++){
	      z3[n][m][r][c]+= w3[n][a][b]*a2[m][2*r + a][2*c + b];
	    }
	  }
	  a3[n][m][r][c]=activation(p[3], z3[n][m][r][c]);
	}
      }
    }
  }

  // apply layer 4
  for (n=0; n<10; n++){
    for (m=0; m<5; m++){
      for (c=0; c<5; c++){
        for (r=0; r<5; r++){
	  a4[n][m][r][c]=0;
          for (b=0; b<3; b++){
	    for (a=0; a<3; a++){
	      if ((r + a < 5) && (c + b < 5)){
                a4[n][m][r][c]+=a3[n][m][r + a][c + b]/(float)9;
	      }
	    }
	  }
	}
      }
    }
  }
  
  // apply layer 5
  for (j=0; j<100; j++){
    z5[j]=b5[j];
    for (n=0; n<10; n++){
      for (m=0; m<5; m++){
        for (c=0; c<5; c++){
	  for (r=0; r<5; r++){
            z5[j]+=w5[j][n][m][r][c]*a4[n][m][r][c];
	  }
	}
      }
    }
    a5[j]=activation(p[5], z5[j]);
  }

  // apply layer 6
  for (j=0; j<10; j++){
    z6[j]=b6[j];
    for (k=0; k<100; k++){
      z6[j]+=w6[j][k]*a5[k];
    }
    a6[j]=activation(p[6], z6[j]);
  }
}


/*
Introduces basic concepts in SEAL and shows how to perform simple 
arithmetic operations on encrypted data.
*/
void example_basics_i();

int main(int argc, char* argv[])
{
/*
Only available when SEAL built using CMake.
*/
#ifdef SEAL_VERSION
    cout << "SEAL version: " << SEAL_VERSION << endl;
#endif
    example_basics_i();

  // set activation function per layer;
  // 0 => sigmoid;
  // 1 => SoftPlus (approx. to ReLU);
  // 2 => identity;
  // 3 => squaring
  // 7 => squaring scaled to [-8, 8]
  // 8 => squaring scaled to [-300, 300]
  int p[NUM_LAYERS + 1] = {0, 7, 2, 2, 2, 8, 0};

  // activated neurons by layer
  float a0[28][28];
  float a1[5][13][13];
  float a2[5][13][13];
  float a3[10][5][5][5];
  float a4[10][5][5][5];
  float a5[100];
  float a6[10];

  // pre-activated neurons by layer
  float z1[5][13][13];
  float z3[10][5][5][5];
  float z5[100];
  float z6[10];

  // weights and biases by layer
  float w1[5][5][5];
  float w3[10][5][5];
  float b5[100];
  float w5[100][10][5][5][5];
  float	b6[10];
  float w6[10][100];

  // testing data
  FILE *fptr_testing_images = fopen(argv[1],"rb");
  FILE *fptr_testing_labels = fopen(argv[2],"rb");
  
  EncryptionParameters parms;
  parms.set_poly_modulus("1x^2048 + 1");

  unsigned char testing_image[28*28];
  unsigned char testing_label;
  
  int i, j, jj, t;
  
  init_net(w1,
	   w3,
	   b5, w5,
	   b6, w6);

  int m, a, b, n, k;
  
  printf("\nBeginning testing...\n");
  for (i=1; i<=10000; i++){
    
    // load sample testing image and its label
    fread(&testing_image, 784, 1, fptr_testing_images);
    fread(&testing_label, 1, 1, fptr_testing_labels);

    // load sample image into layer 0
    for (j=0; j<28; j++){
      for (jj=0; jj<28; jj++){
        a0[j][jj]=(float)testing_image[jj + j*28]/(float)255;
      }
    }
    
    // run net on sample image
    run_testing_net(p,
	            a0,
	            w1,
	            w3,
	            b5, w5,
		    b6, w6,
	            a6);

    // print label and output layer
    printf("%d    ", testing_label);
    for (j=0; j<10; j++){
      printf("%4.3f    ", a6[j]);
    }
    printf("\n");
  }

  // close infiles
  fclose(fptr_testing_images);
  fclose(fptr_testing_labels);





    return 0;
}

void example_basics_i()
{
    /*
    In this example we demonstrate setting up encryption parameters and other 
    relevant objects for performing simple computations on encrypted integers.

    SEAL uses the Brakerski/Fan-Vercauteren (BFV) homomorphic encryption scheme. 
    We refer to https://eprint.iacr.org/2012/144 for full details on how the BFV 
    scheme works. For better performance, SEAL implements the "FullRNS" 
    optimization of BFV, as described in https://eprint.iacr.org/2016/510.
    */

    /*
    The first task is to set up an instance of the EncryptionParameters class.
    It is critical to understand how these different parameters behave, how they
    affect the encryption scheme, performance, and the security level. There are 
    three encryption parameters that are necessary to set: 

        - poly_modulus (polynomial modulus);
        - coeff_modulus ([ciphertext] coefficient modulus);
        - plain_modulus (plaintext modulus).

    A fourth parameter -- noise_standard_deviation -- has a default value of 3.19 
    and should not be necessary to modify unless the user has a specific reason 
    to and knows what they are doing.

    The encryption scheme implemented in SEAL cannot perform arbitrary computations
    on encrypted data. Instead, each ciphertext has a specific quantity called the
    `invariant noise budget' -- or `noise budget' for short -- measured in bits. 
    The noise budget of a freshly encrypted ciphertext (initial noise budget) is 
    determined by the encryption parameters. Homomorphic operations consume the 
    noise budget at a rate also determined by the encryption parameters. In SEAL 
    the two basic homomorphic operations are additions and multiplications, of 
    which additions can generally be thought of as being nearly free in terms of 
    noise budget consumption compared to multiplications. Since noise budget 
    consumption is compounding in sequential multiplications, the most significant 
    factor in choosing appropriate encryption parameters is the multiplicative 
    depth of the arithmetic circuit that needs to be evaluated. Once the noise 
    budget in a ciphertext reaches zero, it becomes too corrupted to be decrypted. 
    Thus, it is essential to choose the parameters to be large enough to support 
    the desired computation; otherwise the result is impossible to make sense of 
    even with the secret key.
    */
    EncryptionParameters parms;

    /*
    We first set the polynomial modulus. This must be a power-of-2 cyclotomic 
    polynomial, i.e. a polynomial of the form "1x^(power-of-2) + 1". The polynomial
    modulus should be thought of mainly affecting the security level of the scheme;
    larger polynomial modulus makes the scheme more secure. At the same time, it
    makes ciphertext sizes larger, and consequently all operations slower.
    Recommended degrees for poly_modulus are 1024, 2048, 4096, 8192, 16384, 32768,
    but it is also possible to go beyond this. Since we perform only a very small
    computation in this example, it suffices to use a very small polynomial modulus.
    */
    parms.set_poly_modulus("1x^2048 + 1");

    /*
    Next we choose the [ciphertext] coefficient modulus (coeff_modulus). The size 
    of the coefficient modulus should be thought of as the most significant factor 
    in determining the noise budget in a freshly encrypted ciphertext: bigger means
    more noise budget. Unfortunately, a larger coefficient modulus also lowers the
    security level of the scheme. Thus, if a large noise budget is required for
    complicated computations, a large coefficient modulus needs to be used, and the
    reduction in the security level must be countered by simultaneously increasing
    the polynomial modulus. 
    
    To make parameter selection easier for the user, we have constructed sets of 
    largest allowed coefficient moduli for 128-bit and 192-bit security levels
    for different choices of the polynomial modulus. These recommended parameters
    follow the Security white paper at http://HomomorphicEncryption.org. However,
    due to the complexity of this topic, we highly recommend the user to directly 
    consult an expert in homomorphic encryption and RLWE-based encryption schemes 
    to determine the security of their parameter choices. 
    
    Our recommended values for the coefficient modulus can be easily accessed 
    through the functions 
        
        coeff_modulus_128bit(int)
        coeff_modulus_192bit(int)

    for 128-bit and 192-bit security levels. The integer parameter is the degree
    of the polynomial modulus.
    
    In SEAL the coefficient modulus is a positive composite number -- a product
    of distinct primes of size up to 60 bits. When we talk about the size of the 
    coefficient modulus we mean the bit length of the product of the small primes. 
    The small primes are represented by instances of the SmallModulus class; for 
    example coeff_modulus_128bit(int) returns a vector of SmallModulus instances. 
    
    It is possible for the user to select their own small primes. Since SEAL uses
    the Number Theoretic Transform (NTT) for polynomial multiplications modulo the
    factors of the coefficient modulus, the factors need to be prime numbers
    congruent to 1 modulo 2*degree(poly_modulus). We have generated a list of such
    prime numbers of various sizes, that the user can easily access through the
    functions 
    
        small_mods_60bit(int)
        small_mods_50bit(int)
        small_mods_40bit(int)
        small_mods_30bit(int)
    
    each of which gives access to an array of primes of the denoted size. These 
    primes are located in the source file util/globals.cpp.

    Performance is mainly affected by the size of the polynomial modulus, and the
    number of prime factors in the coefficient modulus. Thus, it is important to
    use as few factors in the coefficient modulus as possible.

    In this example we use the default coefficient modulus for a 128-bit security
    level. Concretely, this coefficient modulus consists of only one 54-bit prime 
    factor: 0x3fffffff000001.
    */
    parms.set_coeff_modulus(coeff_modulus_128(2048));

    /*
    The plaintext modulus can be any positive integer, even though here we take 
    it to be a power of two. In fact, in many cases one might instead want it to 
    be a prime number; we will see this in example_batching(). The plaintext 
    modulus determines the size of the plaintext data type, but it also affects 
    the noise budget in a freshly encrypted ciphertext, and the consumption of 
    the noise budget in homomorphic multiplication. Thus, it is essential to try 
    to keep the plaintext data type as small as possible for good performance.
    The noise budget in a freshly encrypted ciphertext is 
    
        ~ log2(coeff_modulus/plain_modulus) (bits)

    and the noise budget consumption in a homomorphic multiplication is of the 
    form log2(plain_modulus) + (other terms).
    */
    parms.set_plain_modulus(1 << 8);

    /*
    Now that all parameters are set, we are ready to construct a SEALContext 
    object. This is a heavy class that checks the validity and properties of 
    the parameters we just set, and performs and stores several important 
    pre-computations.
    */
    SEALContext context(parms);

    /*
    Print the parameters that we have chosen.
    */
    print_parameters(context);

    /*
    Plaintexts in the BFV scheme are polynomials with coefficients integers 
    modulo plain_modulus. To encrypt for example integers instead, one can use 
    an `encoding scheme' to represent the integers as such polynomials. SEAL 
    comes with a few basic encoders:

    [IntegerEncoder]
    Given an integer base b, encodes integers as plaintext polynomials as follows. 
    First, a base-b expansion of the integer is computed. This expansion uses 
    a `balanced' set of representatives of integers modulo b as the coefficients. 
    Namely, when b is odd the coefficients are integers between -(b-1)/2 and 
    (b-1)/2. When b is even, the integers are between -b/2 and (b-1)/2, except 
    when b is two and the usual binary expansion is used (coefficients 0 and 1). 
    Decoding amounts to evaluating the polynomial at x=b. For example, if b=2, 
    the integer 
    
        26 = 2^4 + 2^3 + 2^1
    
    is encoded as the polynomial 1x^4 + 1x^3 + 1x^1. When b=3, 
    
        26 = 3^3 - 3^0 
    
    is encoded as the polynomial 1x^3 - 1. In memory polynomial coefficients are 
    always stored as unsigned integers by storing their smallest non-negative 
    representatives modulo plain_modulus. To create a base-b integer encoder, 
    use the constructor IntegerEncoder(plain_modulus, b). If no b is given, b=2 
    is used.

    [FractionalEncoder]
    The FractionalEncoder encodes fixed-precision rational numbers as follows. 
    It expands the number in a given base b, possibly truncating an infinite 
    fractional part to finite precision, e.g. 
    
        26.75 = 2^4 + 2^3 + 2^1 + 2^(-1) + 2^(-2) 
        
    when b=2. For the sake of the example, suppose poly_modulus is 1x^1024 + 1. 
    It then represents the integer part of the number in the same way as in 
    IntegerEncoder (with b=2 here), and moves the fractional part instead to the 
    highest degree part of the polynomial, but with signs of the coefficients 
    changed. In this example we would represent 26.75 as the polynomial 
    
        -1x^1023 - 1x^1022 + 1x^4 + 1x^3 + 1x^1. 
        
    In memory the negative coefficients of the polynomial will be represented as 
    their negatives modulo plain_modulus.

    [PolyCRTBuilder]
    If plain_modulus is a prime congruent to 1 modulo 2*degree(poly_modulus), the 
    plaintext elements can be viewed as 2-by-(degree(poly_modulus) / 2) matrices
    with elements integers modulo plain_modulus. When a desired computation can be
    vectorized, using PolyCRTBuilder can result in massive performance improvements 
    over naively encrypting and operating on each input number separately. Thus, 
    in more complicated computations this is likely to be by far the most important 
    and useful encoder. In example_batching() we show how to use and operate on 
    encrypted matrix plaintexts.

    For performance reasons, in homomorphic encryption one typically wants to keep
    the plaintext data types as small as possible, which can make it challenging to 
    prevent data type overflow in more complicated computations, especially when 
    operating on rational numbers that have been scaled to integers. When using 
    PolyCRTBuilder estimating whether an overflow occurs is a fairly standard task, 
    as the matrix slots are integers modulo plain_modulus, and each slot is operated 
    on independently of the others. When using IntegerEncoder or FractionalEncoder
    it is substantially more difficult to estimate when an overflow occurs in the 
    plaintext, and choosing the plaintext modulus very carefully to be large enough 
    is critical to avoid unexpected results. Specifically, one needs to estimate how 
    large the largest coefficient in  the polynomial view of all of the plaintext 
    elements becomes, and choose the plaintext modulus to be larger than this value. 
    SEAL comes with an automatic parameter selection tool that can help with this 
    task, as is demonstrated in example_parameter_selection().

    Here we choose to create an IntegerEncoder with base b=2.
    */
    IntegerEncoder encoder(context.plain_modulus());

    /*
    We are now ready to generate the secret and public keys. For this purpose we need
    an instance of the KeyGenerator class. Constructing a KeyGenerator automatically 
    generates the public and secret key, which can then be read to local variables. 
    To create a fresh pair of keys one can call KeyGenerator::generate() at any time.
    */
    KeyGenerator keygen(context);
    PublicKey public_key = keygen.public_key();
    SecretKey secret_key = keygen.secret_key();

    /*
    To be able to encrypt, we need to construct an instance of Encryptor. Note that
    the Encryptor only requires the public key.
    */
    Encryptor encryptor(context, public_key);

    /*
    Computations on the ciphertexts are performed with the Evaluator class.
    */
    Evaluator evaluator(context);

    /*
    We will of course want to decrypt our results to verify that everything worked,
    so we need to also construct an instance of Decryptor. Note that the Decryptor
    requires the secret key.
    */
    Decryptor decryptor(context, secret_key);

    /*
    We start by encoding two integers as plaintext polynomials.
    */
    int value1 = 5;
    Plaintext plain1 = encoder.encode(value1);
    cout << "Encoded " << value1 << " as polynomial " << plain1.to_string() << " (plain1)" << endl;

    int value2 = -7;
    Plaintext plain2 = encoder.encode(value2);
    cout << "Encoded " << value2 << " as polynomial " << plain2.to_string() << " (plain2)" << endl;

    /*
    Encrypting the values is easy.
    */
    Ciphertext encrypted1, encrypted2;
    cout << "Encrypting plain1: ";
    encryptor.encrypt(plain1, encrypted1);
    cout << "Done (encrypted1)" << endl;

    cout << "Encrypting plain2: ";
    encryptor.encrypt(plain2, encrypted2);
    cout << "Done (encrypted2)" << endl;

    /*
    To illustrate the concept of noise budget, we print the budgets in the fresh 
    encryptions.
    */
    cout << "Noise budget in encrypted1: " 
        << decryptor.invariant_noise_budget(encrypted1) << " bits" << endl;
    cout << "Noise budget in encrypted2: " 
        << decryptor.invariant_noise_budget(encrypted2) << " bits" << endl;

    /*
    As a simple example, we compute (-encrypted1 + encrypted2) * encrypted2.
    */

    /*
    Negation is a unary operation.
    */
    evaluator.negate(encrypted1);

    /*
    Negation does not consume any noise budget.
    */
    cout << "Noise budget in -encrypted1: " 
        << decryptor.invariant_noise_budget(encrypted1) << " bits" << endl;

    /*
    Addition can be done in-place (overwriting the first argument with the result,
    or alternatively a three-argument overload with a separate destination variable
    can be used. The in-place variants are always more efficient. Here we overwrite
    encrypted1 with the sum.
    */
    evaluator.add(encrypted1, encrypted2);

    /*
    It is instructive to think that addition sets the noise budget to the minimum
    of the input noise budgets. In this case both inputs had roughly the same
    budget going on, and the output (in encrypted1) has just slightly lower budget.
    Depending on probabilistic effects, the noise growth consumption may or may 
    not be visible when measured in whole bits.
    */
    cout << "Noise budget in -encrypted1 + encrypted2: " 
        << decryptor.invariant_noise_budget(encrypted1) << " bits" << endl;

    /*
    Finally multiply with encrypted2. Again, we use the in-place version of the
    function, overwriting encrypted1 with the product.
    */
    evaluator.multiply(encrypted1, encrypted2);

    /*
    Multiplication consumes a lot of noise budget. This is clearly seen in the
    print-out. The user can change the plain_modulus to see its effect on the
    rate of noise budget consumption.
    */
    cout << "Noise budget in (-encrypted1 + encrypted2) * encrypted2: "
        << decryptor.invariant_noise_budget(encrypted1) << " bits" << endl;

    /*
    Now we decrypt and decode our result.
    */
    Plaintext plain_result;
    cout << "Decrypting result: ";
    decryptor.decrypt(encrypted1, plain_result);
    cout << "Done" << endl;

    /*
    Print the result plaintext polynomial.
    */
    cout << "Plaintext polynomial: " << plain_result.to_string() << endl;

    /*
    Decode to obtain an integer result.
    */
    cout << "Decoded integer: " << encoder.decode_int32(plain_result) << endl;
}
