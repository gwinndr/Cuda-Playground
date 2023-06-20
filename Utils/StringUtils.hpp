#include <sstream>

template <class T>
std::string ArrayToStr(const T* aphArr, int lnLen)
{
	std::stringstream lcRetStr;
	lcRetStr << "[";
	
	for (int i = 0; i < lnLen; ++i)
	{
		lcRetStr << aphArr[i];

		if (i != lnLen - 1)
		{
			lcRetStr << ", ";
		}
	}

	lcRetStr << "]";

	return lcRetStr.str();
}
