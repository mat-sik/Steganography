Examples of usage.

1) Hiding rgb image which will be transformed to greyscale in rgb image.
python3 stegano.py -i -p tester._smool.jpg -pm covid.jpg -n greyscale_image_to_hide -nz rgb_image_with_hidden_image

2) Decoding 1)
python3 stegano.py -dec -p rgb_image_with_hidden_image.png -nz greyscale_image_decoded
 
3) Hiding DNA in rgb image.
python3 stegano.py -d -p covid-19.fasta_.txt -pm covid.jpg -nz rgb_image_with_hidden_dna

4) Decoding 3)
python3 stegano.py -dec -p rgb_image_with_hidden_dna.png -nz greyscale_image_decoded

