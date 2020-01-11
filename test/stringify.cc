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
