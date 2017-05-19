// This file is part of cnn-crf-stereo.
//
// Copyright (C) 2017 Christian Reinbacher <reinbacher at icg dot tugraz dot at>
// Institute for Computer Graphics and Vision, Graz University of Technology
// https://www.tugraz.at/institute/icg/teams/team-pock/
//
// cnn-crf-stereo is free software: you can redistribute it and/or modify it under the
// terms of the GNU Affero General Public License as published by the Free Software
// Foundation, either version 3 of the License, or any later version.
//
// cnn-crf-stereo is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
// FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for more details.
//
// You should have received a copy of the GNU Affero General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

#include "cnpy.h"
#include "utils.h"

void save(iu::TensorGpu_32f &d_tensor, std::string path)
{
	iu::TensorCpu_32f *h_tensor;
	if(d_tensor.memoryLayout() == iu::TensorGpu_32f::MemoryLayout::NHWC)
	{
		h_tensor = new iu::TensorCpu_32f(d_tensor.samples(), d_tensor.height(), d_tensor.width(), d_tensor.channels());
		//h_tensor = new iu::TensorCpu_32f(d_tensor.samples(), d_tensor.channels(), d_tensor.height(), d_tensor.width(), iu::Tensor);
	}
	else
	{
		std::cout << "Use NCHW for output" << std::endl;
		h_tensor = new iu::TensorCpu_32f(d_tensor.samples(), d_tensor.channels(), d_tensor.height(), d_tensor.width());
	}

	iu::copy(&d_tensor, h_tensor);

	const unsigned int shape[] = { h_tensor->samples(), h_tensor->channels(), h_tensor->height(), h_tensor->width() };
	std::cout << "(debug) shape " << shape[0] << " " << shape[1] << " " << shape[2] << " " << shape[3] << std::endl;
	cnpy::npy_save(path, h_tensor->data(), shape, 4, "w");
	std::cout << "save done: " << path << std::endl;

	delete h_tensor;

}
