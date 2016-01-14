#include <algorithm>
#include <cstdlib>

//A simple un-optimized reference radix sort calculation
//Only deals with power-of-2 radices


void reference_calculation(unsigned int* inputVals,
                           unsigned int* inputPos,
                           unsigned int* outputVals,
                           unsigned int* outputPos,
                           const size_t numElems)
{
  const int numBits = 1;
  const int numBins = 1 << numBits;

  unsigned int *binHistogram = new unsigned int[numBins];
  unsigned int *binScan      = new unsigned int[numBins];

  unsigned int *vals_src = inputVals;
  unsigned int *pos_src  = inputPos;

  unsigned int *vals_dst = outputVals;
  unsigned int *pos_dst  = outputPos;

  //a simple radix sort - only guaranteed to work for numBits that are multiples of 2
// 8 bits per byte!
  for (unsigned int i = 0; i < 8 * sizeof(unsigned int); i += numBits) {
    unsigned int mask = (numBins - 1) << i;
//  (2 - 1) << 0 = 0000 0001
//  (2 - 1) << 1 = 0000 0010
//  ...

    memset(binHistogram, 0, sizeof(unsigned int) * numBins); //zero out the bins
    memset(binScan, 0, sizeof(unsigned int) * numBins); //zero out the bins

    //perform histogram of data & mask into bins
    for (unsigned int j = 0; j < numElems; ++j) {
      unsigned int bin = (vals_src[j] & mask) >> i;
      binHistogram[bin]++;
//    i = 0        i = 1        i = 2
//    0000 0001    0000 0010    0000 0100
//  & 0000 0001    1111 0011    1111 1011
//  = 0000 0001    0000 0010    0000 0000
//  >> i
//    0000 0001    0000 0001    0000 0000
//    WILL PLACE IN HISTOGRAM ACCORDING TO LSB!
    }

    //perform exclusive prefix sum (scan) on binHistogram to get starting
    //location for each bin
    for (unsigned int j = 1; j < numBins; ++j) {
      binScan[j] = binScan[j - 1] + binHistogram[j - 1];
//  j = 1
//  binScan[1] + binHistogram[0] = 0 + AMOUNT_OF_DIGITS_IN_BIN_HISTO
//  GIVES SPLITTER LOCATION FOR 0 AND 1 LSB!
    }

    //Gather everything into the correct location
    //need to move vals and positions
    for (unsigned int j = 0; j < numElems; ++j) {
// bin = 1 / 0
      unsigned int bin = (vals_src[j] & mask) >> i;
// MOVE EACH ELEMENT TO ITS CORRESPONDING LOCATION
      vals_dst[binScan[bin]] = vals_src[j];
      pos_dst[binScan[bin]]  = pos_src[j];
// UPDATE SPLITTER IN BINsCAN FOR NEXT ITERATION
      binScan[bin]++;
    }

    //swap the buffers (pointers only)
    std::swap(vals_dst, vals_src);
    std::swap(pos_dst, pos_src);
  }

  //we did an even number of iterations, need to copy from input buffer into output
// (SRC.FIRST, SRC.LAST, DST)
  std::copy(inputVals, inputVals + numElems, outputVals);
  std::copy(inputPos, inputPos + numElems, outputPos);

  delete[] binHistogram;
  delete[] binScan;
}

