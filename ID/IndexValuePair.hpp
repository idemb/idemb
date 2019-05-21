#pragma once

#include <iostream>

struct IndexValuePair {

	size_t m_index; 
	double m_value;
	
	IndexValuePair(size_t i= 0, double v=0.0) : m_index(i), m_value(v) { }
	
	friend std::ostream& operator<<(std::ostream& os, const IndexValuePair& pair)  {
        return os << "index = " << pair.m_index << " value = " << pair.m_value << "\n";
	}
    
	bool operator==(const IndexValuePair& other) const {
		return ((m_index == other.m_index) && (m_value == other.m_value));
    }
	
};
