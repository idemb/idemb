#ifndef _SERVICES_H
#define _SERVICES_H

#include <iostream>
#include <fstream>
#include <stdio.h>
#include <vector>

template <typename T> void print(std::ostream& os, const std::vector<T> arg)
{
	for (size_t i=0;i<arg.size(); i++){
		os << " " << arg[i] ;
	}
}


template <typename T> std::ostream& operator<<(std::ostream& os, std::vector<T>& arg)
{
    print(os, arg);
    return os;
}


#endif
