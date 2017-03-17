#pragma once
#include <vector>

#include "operator.h"

class SlackProp: public Operator
{
public:
	SlackProp(cudnnTensorDescriptor_t inTensorDesc);
	~SlackProp();

	// managed methods
	iu::TensorGpu_32f *forward(std::vector<iu::TensorGpu_32f *> &d_inputs,
			cudnnHandle_t cudnnHandle);

	cudnnTensorDescriptor_t outTensorDesc();

	void setLambda(float value);
	void setP1(float value);
	void setP2(float value);

	// output is float again
	static float forward(iu::TensorGpu_32f * costVolume , iu::TensorGpu_32f * pwVolume, iu::TensorGpu_32f *output, int max_iter=5);

	// old interface, only wx and wy
	static float forward(float *costVolume, float *wx, float *wy, float *output, int in, int ic,
				int ih, int iw, float P1, float P2, int max_iter=5);

	static float forwardVolumeOut(float *costVolume, float *wx, float *wy, float *output, int in, int ic,
			int ih, int iw, float P1, float P2, int max_iter, int delta);

private:
	// no copies!
	SlackProp(SlackProp const&);
	SlackProp& operator=(SlackProp const&);

	float m_P1;
	float m_P2;
	float m_lamda;

	float m_lowerBound;

	cudnnTensorDescriptor_t m_outTensorDesc;
	iu::TensorGpu_32f *m_output;

};
