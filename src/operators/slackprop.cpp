// This file is part of cnn-crf-stereo.
//
// Copyright (C) 2017 Patrick Knöbelreiter <knoebelreiter at icg dot tugraz dot at>
// Institute for Computer Graphics and Vision, Graz University of Technology
// https://www.tugraz.at/institute/icg/teams/team-pock/
//
// dvs-panotracking is free software: you can redistribute it and/or modify it under the
// terms of the GNU Affero General Public License as published by the Free Software
// Foundation, either version 3 of the License, or any later version.
//
// cnn-crf-stereo is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
// FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for more details.
//
// You should have received a copy of the GNU Affero General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

#include "slackprop.h"

// Sashas algorithm
#include "slack_prop.h"
#include "iu/ndarray/ndarray.h"
#include "iu/ndarray/ndarray_iu.h"

SlackProp::SlackProp(cudnnTensorDescriptor_t inTensorDesc) :
Operator(inTensorDesc, Operator::SLACKPROP), m_P1(1.0), m_P2(7.0), m_lamda(0.18), m_lowerBound(
		-1.0)
{

}

SlackProp::~SlackProp()
{

}

cudnnTensorDescriptor_t SlackProp::outTensorDesc()
{
	return m_outTensorDesc;
}

void SlackProp::setLambda(float value)
{
	m_lamda = value;
}

void SlackProp::setP1(float value)
{
	m_P1 = value;
}

void SlackProp::setP2(float value)
{
	m_P2 = value;
}

iu::TensorGpu_32f *SlackProp::forward(std::vector<iu::TensorGpu_32f *> &d_inputs,
		cudnnHandle_t cudnnHandle)
{
//	iu::TensorGpu_32f *costVolume = d_inputs[0];
//	iu::TensorGpu_32f *wx = d_inputs[1];
//	iu::TensorGpu_32f *wy = d_inputs[2];
//
//	forward(costVolume->data(), wx->data(), wy->data(), m_output->data(), costVolume->samples(),
//			costVolume->channels(), costVolume->height(), costVolume->width(), m_P1, m_P2);
//
	return NULL;
}

// attention! this will only work with tensor layout NHWC
float SlackProp::forward(iu::TensorGpu_32f * costVolume , iu::TensorGpu_32f * pwVolume, iu::TensorGpu_32f *output, int max_iter)
{
	//int K = ic;
	auto Cs =  costVolume->ref().swap_dims(0,3).subdim<3>(0); // 1 x D1 x D2 x K     (N H W C)

	int D1 = Cs.size(1);
	int D2 = Cs.size(2);

	auto W_in = pwVolume->ref().subdim<0>(0); // 1 x (P1x, P1y, P2x, P2y, .. P3x P3y) x D1 x D2     (N C H W)
	int pw_channels = W_in.size(0) / 2;
	runtime_check(D1 == W_in.size(1));
	runtime_check(D2 == W_in.size(2));

	ndarray_ref<float,2> sols = output->ref().subdim<0,1>(0,0); // 1 x 1 x D1 x D2     (N C H W)
	runtime_check(D1 == sols.size(0));
	runtime_check(D2 == sols.size(1));

//	std::cout <<"Cs=" << Cs<<"\n";
//	std::cout <<"W_in="<< W_in<<"\n";
//	std::cout <<"sols="<< sols<<"\n";

	ndarray<float,4> W;
	W.create<memory::GPU>({pw_channels, D1, D2, 2});

	/*
	for(int i = 0; i <  pw_channels*2; ++i){
		//int d = i % 2;
		int d = 1 - (i % 2);
		int c = i / 2;
	//	W [channels x image_size0 x image_size1 x n_dirs]
		if(c== 0){
			W.subdim<0,3>(c,d) << W_in.subdim<0>(i);
		}else{// c = 1
			( W.subdim<0,3>(c,d) << W_in.subdim<0>(i) ) -=  ;
		};
	};
	*/

	//! rearrange W_in as [(P1,P2) x D1 x D2 x (x,y)]
	W << W_in.reshape(2, pw_channels, D1, D2).flip_dim(1).permute_dims({1,2,3,0});

	/*
	slack_prop_ops ops;

	for(int d =0; d <2; ++d){
		auto P1 = W_in.subdim<0>(d);
		auto P2 = W_in.subdim<0>(d + 2);

		W.subdim<0,3>(0,1-d) << P1;
		//((W.subdim<0,3>(1,1-d) << P2) -= P1)*=(1.0f/(alg.ops.delta-1));
		float c = 1.0f/(ops.delta-1);
		madd2(W.subdim<0,3>(1,1-d), P1, P2, -c, c);
	};
	*/

	ndarray_ref<int,2> sols_i = sols.recast<int>();

	float LB;
	{
		slack_prop_2D_alg alg;
		alg.ops["total_it"] = max_iter;
		alg.init(Cs, sols_i);
		alg.update_W(W);
		alg.execute();
		sols << sols_i; // call kernel to convert int to float
		LB = alg.LB();
	}

	return LB;
}

// attention! this will only work with tensor layout NHWC // OLD INTERFACE
float SlackProp::forward(float *costVolume, float *wx, float *wy, float *output, int in, int ic,
		int ih, int iw, float P1, float P2, int max_iter)
{
	int K = ic;
	int D1 = iw;
	int D2 = ih;

	ndarray_ref<float,3> Cs(costVolume,{K,D1,D2},ndarray_flags::device_only);
	ndarray_ref<float,2> w0(wx, {D1,D2},ndarray_flags::device_only);
	ndarray_ref<float,2> w1(wy, {D1,D2},ndarray_flags::device_only);
	ndarray_ref<float,2> sols(output, {D1,D2},ndarray_flags::device_only);
	ndarray_ref<int,2> sols_i = sols.recast<int>();

	float LB;
	{
		slack_prop_2D_alg alg;
		alg.ops["L1"] = P1;
		alg.ops["L2"] = P2;
		alg.ops["delta"] = 2;
		alg.ops["total_it"] = max_iter;
		alg.init(Cs, sols_i);
		alg.update_W(w0,w1);
		alg.execute();
		sols << sols_i;
		LB = alg.LB();
	}

	return LB;
}

// attention! this will only work with tensor layout NHWC // OLD INTERFACE
float SlackProp::forwardVolumeOut(float *costVolume, float *wx, float *wy, float *output, int in, int ic,
		int ih, int iw, float P1, float P2, int max_iter, int delta)
{
	int K = ic;
	int D1 = iw;
	int D2 = ih;

	ndarray_ref<float,3> Cs(costVolume,{K,D1,D2},ndarray_flags::device_only);
	ndarray_ref<float,2> w0(wx, {D1,D2},ndarray_flags::device_only);
	ndarray_ref<float,2> w1(wy, {D1,D2},ndarray_flags::device_only);

	iu::TensorGpu_32s iu_sols_i(1, 1, D1, D2);
	ndarray_ref<int,2> sols_i = iu_sols_i.ref().subdim<0>(0).subdim<0>(0);

	ndarray_ref<float,3> cv_reg(output, {K, D1, D2},ndarray_flags::device_only);

	float LB;
	{
		slack_prop_2D_alg alg;
		alg.ops["L1"] = P1;
		alg.ops["L2"] = P2;
		alg.ops["delta"] = delta;
		alg.ops["total_it"] = max_iter;
		alg.init(Cs, sols_i);
		alg.update_W(w0, w1);
		alg.execute();
		cv_reg << alg.get_C1();
		LB = alg.LB();
	}

	return LB;
}
