// ============================================================================
// data_utils.h - File I/O utilities for Ascend C Kernel direct invocation
// ============================================================================

#ifndef DATA_UTILS_H
#define DATA_UTILS_H

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cerrno>
#include <fstream>

inline bool ReadFile(const std::string &filePath, size_t bufferSize, void *buffer, size_t bufferLen)
{
    if (buffer == nullptr) {
        return false;
    }
    if (bufferSize > bufferLen) {
        return false;
    }

    std::ifstream file(filePath, std::ios::binary);
    if (!file.is_open()) {
        return false;
    }

    file.seekg(0, std::ios::end);
    auto pos = file.tellg();
    if (pos < 0) {
        file.close();
        return false;
    }
    size_t fileSize = static_cast<size_t>(pos);
    file.seekg(0, std::ios::beg);

    if (fileSize != bufferSize) {
        file.close();
        return false;
    }

    file.read(static_cast<char *>(buffer), bufferSize);
    if (!file) {
        file.close();
        return false;
    }

    file.close();
    return true;
}

inline bool WriteFile(const std::string &filePath, const void *buffer, size_t size)
{
    if (buffer == nullptr && size > 0) {
        return false;
    }

    int fd = open(filePath.c_str(), O_RDWR | O_CREAT | O_TRUNC, S_IRUSR | S_IWUSR);
    if (fd < 0) {
        return false;
    }

    size_t totalWritten = 0;
    while (totalWritten < size) {
        ssize_t w = write(fd, static_cast<const char *>(buffer) + totalWritten, size - totalWritten);
        if (w < 0) {
            if (errno == EINTR) continue;
            close(fd);
            return false;
        }
        if (w == 0) {
            close(fd);
            return false;
        }
        totalWritten += static_cast<size_t>(w);
    }
    if (close(fd) != 0) {
        return false;
    }

    return true;
}

#endif
