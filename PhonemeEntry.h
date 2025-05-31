#pragma once
#include <string>
#include <sstream>
#include <vector>
#include <stdexcept>

struct PhonemeEntry {
    std::string waveFilePath;
    std::string alias;
    double offset = 0.0;
    double consonant = 0.0;
    double cutoff = 0.0;
    double preUtterance = 0.0;
    double overlap = 0.0;

    PhonemeEntry() = default;

    PhonemeEntry(const std::string& wave, const std::string& ali, double off, double cons, double cut, double pre, double over)
        : waveFilePath(wave), alias(ali), offset(off), consonant(cons), cutoff(cut), preUtterance(pre), overlap(over) {}

    // Parse a line from oto.ini and return a PhonemeEntry
    static PhonemeEntry parseOtoLine(const std::string& line) {
        // Format: wavFilePath=alias,offset,consonant,cutoff,preutterance,overlap
        auto eqPos = line.find('=');
        if (eqPos == std::string::npos)
            throw std::runtime_error("Invalid oto.ini line: missing '='");

        std::string wave = line.substr(0, eqPos);
        std::string rest = line.substr(eqPos + 1);

        std::vector<std::string> fields;
        std::stringstream ss(rest);
        std::string item;
        while (std::getline(ss, item, ',')) {
            fields.push_back(item);
        }
        if (fields.size() != 6)
            throw std::runtime_error("Invalid oto.ini line: incorrect number of fields");

        PhonemeEntry entry;
        entry.waveFilePath = wave;
        entry.alias = fields[0];
        entry.offset = std::stod(fields[1]);
        entry.consonant = std::stod(fields[2]);
        entry.cutoff = std::stod(fields[3]);
        entry.preUtterance = std::stod(fields[4]);
        entry.overlap = std::stod(fields[5]);
        return entry;
    }
};
