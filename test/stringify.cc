//
// Copyright (C) 2010-2025 Axel Zeuner
//
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
//
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA
//
#include <iostream>
#include <string>

int main()
{
    std::string l;
    bool first=true;
    std::cout << "\"";
    while (std::getline(std::cin, l).good()) {
        if (first == false) {
            std::cout <<"\n\"";
        } else {
            first = false;
        }
        for (std::size_t i=0; i<l.size(); ++i) {
            std::string::value_type c=l[i];
            switch (c) {
            case '\t':
                std::cout << "\\t";
                break;
            case '\n':
                std::cout << "\\n";
                break;
            case '"':
                std::cout << "\\\"";
                break;
            default:
                std::cout << c;
                break;
            }
        }
        std::cout <<"\\n\"";
    }
    std::cout << ";\n";
    return 0;
}
