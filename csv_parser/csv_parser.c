/**
 * This is the actual implementation of the csv parser API exposed in
 * csv_parser.h
 *
 * The current implementation is a revision of a previous implementation, made
 * with the assistance of Gemini 2.5 Pro LLM model.
 *
 * Since I am a novice in C, my initial implementation was inefficient, and
 * I wanted to make it faster to work easily even with very large csv files.
 * Thanks to the suggestions made by the model, I was able to reduce the parsing
 * time required for a 10GB csv file with 1,000,000 entries and 1,000 columns
 * from 43 seconds to 10 seconds (if compiled with -O3 flag).
 *
 * The primary ineffiency was due to the high amount of malloc that I performed
 * in the original implementation.
 */

#include "csv_parser.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <sys/stat.h>

// Function that creates the parser object and initializes it.
CsvParser* csv_parser_create(char *filename, char delimiter, int has_header) {
    CsvParser *parser = malloc(sizeof(CsvParser));

    // strdup function from string.h saves a copy of the string.
    // It allocates space on the heap using malloc, does the copy and returns
    // a pointer to the copy.
    parser->filename = strdup(filename);
    parser->delimiter = delimiter;
    parser->has_header = has_header;

    // We initialize these attributes to default values to avoid
    // possible unexpected problems with previous values in memory.
    parser->data = NULL;
    parser->buffer = NULL;
    parser->errorMessage = NULL;
    parser->ref_counter = NULL;

    parser->num_cols = 0;
    parser->num_rows = 0;

    return parser;
}

// Function that returns an element of the file.
char *csv_parser_get_field(CsvParser *parser, int row, int col) {
    return parser->data[row][col];
}

// Helper function that returns the size of the csv file to be parsed.
//
// It uses the stat() system call to obtain information about the file
// pointed by path and populates the stat struct.
long get_file_size(const char *filename) {
    struct stat st;
    if (stat(filename, &st) == 0) {
        return st.st_size;
    }

    return -1;
}

// Function that parses the csv file.
//
// The original implementation of the parse function allocated memory for
// each row and each row element separately. This caused a major bottleneck,
// since the number of malloc was proportional to the number of elements
// of the csv file. Furthermore, the file was read line by line, causing further
// reduction in performance.
//
// To reduce parsing time, we allocate a buffer that can contain the entire file
// and we read it altogether. Then, we iterate over the file content
// to get the number of columns and rows,
// and allocate a single continuous memory block for all the fields.
// Finally, we iterate over the buffer again to fill the fields' memory block
// with the actual data.
//
// With this new approach, we now perform only three major malloc operations
// drastically reducing the time required for performing the parsing operation.
int csv_parser_parse(CsvParser *parser) {
    // Opening file and checking for possible errors.
    FILE *file = fopen(parser->filename, "r");
    if (file == NULL) {
        parser->errorMessage = "Failed to open file";
        return 1;
    }

    // Getting file size and checking for possible errors.
    long file_size = get_file_size(parser->filename);
    if (file_size == -1) {
        parser->errorMessage = "Failed to get file size";
        fclose(file);
        return 1;
    }

    // Here we allocate the buffer for the entire file and we read it
    // using fread.
    //
    // fread() allows us to read a binary stream of input. In this case we read
    // a stream of input the size of the file.
    char *buffer = malloc(file_size + 1);

    // If there are errors in the buffer creation or in the reading we
    // return.
    if (!buffer) {
        parser->errorMessage = "Malloc failed for file content";
        fclose(file);
        return 1;
    }

    // fread returns the number of objects written, therefore we check if it is
    // equal to the number of bytes of the file, since we are writing a sequence
    // of chars (usually one byte long), to look for errors.
    if (fread(buffer, 1, file_size, file) != file_size) {
        parser->errorMessage = "Failed to read entire file";
        free(buffer);
        fclose(file);
        return 1;
    }

    // We append the terminator to the buffer to make it act as a string.
    buffer[file_size] = '\0';

    fclose(file);

    // Scan to get the number of rows and columns.
    int rows = 0;
    int cols = 0;
    for (long i = 0; i < file_size; ++i) {
        if (buffer[i] == '\n') {
            rows++;
        } else if (rows == 0 && buffer[i] == parser->delimiter) {
            // We check that rows == 0 to perform the column count just one time
            cols++;
        }
    }

    if (file_size > 0 && buffer[file_size - 1] != '\n') {
        rows++;
    }
    cols++; // There's one more column than delimiters (e.g. col1,col2,col3)

    parser->num_rows = rows;
    parser->num_cols = cols;

    // We allocate the memory for the data attribute of parser and for all
    // the file fields.
    //
    // We allocate a contiguous block of memory for all the fields in order to
    // improve cache locality.
    parser->data = malloc(parser->num_rows * sizeof(char**));
    char **fields = malloc(parser->num_rows * parser->num_cols * sizeof(char*));

    if (!parser->data || !fields) {
        parser->errorMessage = "Malloc failed for data pointers";
        free(buffer);
        free(parser->data);
        free(fields);
        return 1;
    }

    // Link the row pointers to the contiguous field block.
    //
    // This way we can access memory as a matrix even though it is continuous.
    // Moreover we would be able to free fields memory without storing
    // the pointer in the struct, as it will be contained in parser->data[0].
    for (int i = 0; i < parser->num_rows; ++i) {
        parser->data[i] = fields + i * parser->num_cols;
    }

    // Perform the actual parsing.
    //
    // To perform parsing we just make each parser->data[i][j] to the buffer
    // point where each field starts. This way we do not have to copy memory and
    // reduce parsing time.
    char *current_pos = buffer;
    for (int i = 0; i < parser->num_rows; ++i) {
        for (int j = 0; j < parser->num_cols; ++j) {
            parser->data[i][j] = current_pos;

            // We make the pos advance until we find a delimiter or a newline.
            while (*current_pos && *current_pos != parser->delimiter && *current_pos != '\n') {
                current_pos++;
            }

            // If we found a separator, null-terminate the string
            // and move past it.
            if (*current_pos) {
                *current_pos = '\0';
                current_pos++;
            }
        }
    }

    // We save the buffer into the parser so that we will be able to free it
    // later.
    parser->buffer = buffer;

    parser->ref_counter = malloc(sizeof(int));
    *(parser->ref_counter) = 1;

    return 0;
}

// This function allow the user to horizontally split the csv content.
//
// It can be useful when dealing with some datasets that have to be split
// into training and testing.
//
// E.g.
//
// 5.1,3.5,1.4,0.2,Iris-setosa
// 4.9,3.0,1.4,0.2,Iris-setosa
// 4.7,3.2,1.3,0.2,Iris-setosa
// --------------------------- We split the dataset here
// 4.6,3.1,1.5,0.2,Iris-setosa
// 5.0,3.6,1.4,0.2,Iris-setosa
// 5.4,3.9,1.7,0.4,Iris-setosa
//
// To make the split operation fast, we simply work with pointers
// without making any actual copy of the data.
// Specifically, we create a new parser object with the same data
// from the original parser and,
// then, we make the new parser to point towards the rows of the old parser.
// Finally, we reduce the number of columns of the original parser.
//
// This approach is simple to implement but has some flaws, because if we want
// to destroy the original parser but to use the new one, we will erase
// the memory allocated and that will lead to a segmentation fault since we will
// free the memory used by the new parser. To avoid this we introduced,
// reference pointing and we increment the counter every time we split.
// The counter is managed using an integer pointer, so that it can be shared
// among all the parsers objects that refer to the same portion of memory.
CsvParser* csv_parser_split(CsvParser *parser, int row) {
    // Creation of the new parser and set up of its attributes
    CsvParser *new_parser = csv_parser_create(
        parser->filename, parser->delimiter, 0
    );

    new_parser->data = malloc(parser->num_rows * sizeof(char**));
    new_parser->num_cols = parser->num_cols;
    new_parser->num_rows = parser->num_rows - row;
    new_parser->buffer = parser->buffer;

    // Increasing reference counter to take multiple reference into account
    new_parser->ref_counter = parser->ref_counter;
    *(new_parser->ref_counter) += 1;

    // Linking new parser row pointers to the correspondent
    // parser row pointers.
    for (int i = 0, j = row; i < new_parser->num_rows; i++, j++) {
        new_parser->data[i] = parser->data[j];
    }

    parser->num_rows = row;

    return new_parser;
}

// Function that frees the memory allocated
//
// The function take into account the possibility
// of multiple references. So, it checks if the reference
// counter is lower or equal to one, otherwise it will just
// decrement the counter and free parser->data.
void csv_parser_destroy(CsvParser *parser) {
    if (parser == NULL) return;

    if (*(parser->ref_counter) <= 1) {
        free(parser->filename);
        free(parser->buffer);
        free(parser->ref_counter);

        if (parser->data != NULL) free(parser->data[0]);
    } else {
        *(parser->ref_counter) -= 1;
    }

    free(parser->data);
    free(parser);
}
