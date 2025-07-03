/**
 * This header file exposes the API for a simple csv file parser.
 */

typedef struct {
    char *filename;
    char delimiter;
    int has_header;

    char ***data;
    int num_rows;
    int num_cols;

    char *errorMessage;
    char *buffer;
    int *ref_counter;
} CsvParser;

CsvParser* csv_parser_create(char *filename, char delimiter, int has_header);
char *csv_parser_get_field(CsvParser *parser, int i, int j);

int csv_parser_parse(CsvParser *parser);

CsvParser* csv_parser_split(CsvParser *parser, int row);

void csv_parser_destroy(CsvParser *parser);
