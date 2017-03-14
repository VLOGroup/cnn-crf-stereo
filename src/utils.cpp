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
