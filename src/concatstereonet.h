#pragma once
#include "colorstereonet.h"

class ConcatStereoNet : public ColorStereoNet
{
  public:
	ConcatStereoNet(unsigned int numLayers, unsigned int in, unsigned int ic, unsigned int ih, unsigned int iw);
	~ConcatStereoNet();


//private:
	virtual void initNet(float min_disp=0, float max_disp=0, float step=0, int rect_corr=0,
						iu::TensorGpu_32f *d_unaryOut=NULL,	iu::TensorGpu_32f *d_pairwiseOut=NULL);

	iu::TensorGpu_32f	*m_d_outLeft;
	iu::TensorGpu_32f	*m_d_outRight;

	std::vector<iu::TensorGpu_32f*> m_d_leftLayerOutputs;
	std::vector<iu::TensorGpu_32f*> m_d_rightLayerOutputs;

	int m_growthRate;
};
