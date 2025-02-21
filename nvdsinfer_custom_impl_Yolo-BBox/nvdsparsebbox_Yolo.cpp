#include <algorithm>  // For standard algorithms like min and max
#include <cassert>    // For assert function, checking conditions at runtime
#include <cmath>      // For math functions like clamp
#include <cstring>    // For string manipulation functions
#include <fstream>    // For file operations
#include <iostream>   // For input/output operations
#include <unordered_map> // For unordered_map container
#include "nvdsinfer_custom_impl.h"  // For custom inference implementation in NVIDIA DeepStream
#include "gstnvdsmeta.h"  // For DeepStream metadata handling

// #define MIN(a,b) ((a) < (b) ? (a) : (b)) // Define macro for minimum value between a and b
// #define MAX(a,b) ((a) > (b) ? (a) : (b)) // Define macro for maximum value between a and b
#define CLIP(a,min,max) (MAX(MIN(a, max), min))  // Define macro to clip value 'a' within min and max
#define DIVIDE_AND_ROUND_UP(a, b) ((a + b - 1) / b) // Macro to divide and round up

static const int NUM_CLASSES_YOLO = 80; // Number of object classes in YOLO model

// Function to clamp the value 'val' between 'minVal' and 'maxVal'
float clamp(const float val, const float minVal, const float maxVal)
{
    assert(minVal <= maxVal);  // Ensure that minVal is not greater than maxVal
    return std::min(maxVal, std::max(minVal, val));  // Return clamped value
}

// Prototype for custom YOLO NMS parsing function
extern "C" bool NvDsInferParseCustomYolorNms(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferParseObjectInfo>& objectList); 

// Custom NMS function for YOLO detection
extern "C" bool NvDsInferParseCustomYolorNms(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferParseObjectInfo>& objectList)
{
    std::vector<NvDsInferParseObjectInfo> objects; // Vector to store detected objects
    const NvDsInferLayerInfo &output_nms = outputLayersInfo[0]; // Get NMS output layer info
    assert(output_nms.inferDims.numDims == 2);  // Assert that the NMS output has 2 dimensions
    int numberDims = output_nms.inferDims.numDims; // Get number of dimensions

    // Create a vector to store the size of each dimension
    std::vector<unsigned int> dimension_size(numberDims);  
    for (unsigned int i = 0; i < numberDims; i++) {
        dimension_size[i] = output_nms.inferDims.d[i];  // Fill in the dimension sizes
    }
    
    std::vector<NvDsInferParseObjectInfo> outObjs; // Vector to hold the output object information
    float netW = networkInfo.width;  // Network width
    float netH = networkInfo.height;  // Network height
    
    const float* element_output_nms = (const float*) output_nms.buffer;  // Pointer to NMS output buffer
    uint element_nms_location = 0; // Index for navigating through NMS output
    for (uint b = 0; b < dimension_size[0]; ++b)  // Loop through all detections
    {
        // Extract bounding box coordinates and probabilities from the NMS output
        float bx1 = element_output_nms[element_nms_location];
        float by1 = element_output_nms[element_nms_location + 1];
        float bx2 = element_output_nms[element_nms_location + 2];
        float by2 = element_output_nms[element_nms_location + 3];
        float maxProb = element_output_nms[element_nms_location + 4];
        int maxIndex = (int) element_output_nms[element_nms_location + 5];

        // Check if the detection confidence is above threshold
        if (maxProb > detectionParams.perClassPreclusterThreshold[maxIndex])
        {
            NvDsInferParseObjectInfo bbi;  // Object info structure for bounding box
            
            // Restore coordinates to network input resolution
            float x1 = clamp(bx1, 0, netW-1);
            float y1 = clamp(by1, 0, netH-1);
            float x2 = clamp(bx2, 0, netW-1);
            float y2 = clamp(by2, 0, netH-1);
        
            bbi.left = x1;  // Set the left coordinate of bounding box
            bbi.width = clamp(x2 - x1, 0, netW-1);  // Set the width of bounding box
            bbi.top = y1;   // Set the top coordinate of bounding box
            bbi.height = clamp(y2 - y1, 0, netH-1);  // Set the height of bounding box

            // Add the object info to the output list if the bounding box has valid size
            if (bbi.width > 1 || bbi.height > 1) {
                bbi.detectionConfidence = maxProb;  // Set detection confidence
                bbi.classId = maxIndex;  // Set detected class ID
                outObjs.push_back(bbi);  // Add object to output list
            }
        }

        element_nms_location += 6;  // Move to the next element in NMS output
    }

    // Add the detected objects to the final list of objects
    objects.insert(objects.end(), outObjs.begin(), outObjs.end());

    objectList = objects;  // Assign detected objects to output parameter

    return true;  // Return true to indicate successful parsing
}

CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseCustomYolorNms);  // Check prototype for the custom parse function
