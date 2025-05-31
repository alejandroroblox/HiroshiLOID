#pragma once
#include "PhonemeEntry.h"
#include <map>
#include <fstream>
#include <iostream>

class Voicebank {
public:
    // Map from alias to PhonemeEntry
    std::map<std::string, PhonemeEntry> phonemeMap;

    // Load oto.ini from path
    void load(const std::string& otoIniPath) {
        std::ifstream file(otoIniPath);
        if (!file.is_open()) {
            throw std::runtime_error("Failed to open oto.ini: " + otoIniPath);
        }
        std::string line;
        size_t lineNum = 0;
        while (std::getline(file, line)) {
            ++lineNum;
            if (line.empty() || line[0] == '#') continue; // skip comments and empty lines
            try {
                PhonemeEntry entry = PhonemeEntry::parseOtoLine(line);
                phonemeMap[entry.alias] = entry;
            } catch (const std::exception& e) {
                std::cerr << "Error parsing oto.ini line " << lineNum << ": " << e.what() << std::endl;
            }
        }
    }

    // Get PhonemeEntry by alias
    const PhonemeEntry* getPhonemeEntry(const std::string& alias) const {
        auto it = phonemeMap.find(alias);
        if (it != phonemeMap.end()) {
            return &(it->second);
        }
        return nullptr;
    }
};
