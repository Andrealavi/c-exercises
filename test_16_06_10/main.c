/**
 * This test was about creating a small file for managing umbrellas renting
 * on a beach.
 *
 * The beach is represented as a grid of values of the following type:
 *
 * O F O
 * O O O
 * F F F
 *
 * where O stands for Occupied, whereas F stands for Free
 */

#include <stdbool.h>
#include <stdio.h>
#include <string.h>

// Beach dimensions defined at compile-time
#define N 3
#define M 3

// Prints the state of the beach
void print_state(bool beach[N][M]) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            beach[i][j] ? printf(" O ") : printf(" F ");
        }

        printf("\n");
    }
}

// Function for renting an umbrella on the beach.
//
// It gets the row where the user wants to rent the umbrella,
// if there is a free umbrella on the row it is rented
int rent_umbrella(bool beach[N][M], const int row) {
    for (int j = 0; j < M; j++) {
        if (!beach[row - 1][j]) {
            printf("Successfully rented an umbrella.\n");

            beach[row - 1][j] = true;
            return j;
        }
    }

    printf("Rent wasn't possible. The row is completely full.\n");
    return -1;
}

// Helper function used to check if the adjacent positions are free
// or not.
//
// This function is used by rent_isolated_umbrella function
bool check_adjacent(bool beach[N][M], int i, int j) {
    int n, s, e, w;
    bool adjacency = true;

    n = i - 1;
    s = i + 1;
    e = j + 1;
    w = j - 1;

    if (n >= 0) adjacency = !beach[n][j];
    if (s < N) adjacency = adjacency && !beach[s][j];
    if (w >= 0) adjacency = adjacency && !beach[i][w];
    if (e < M) adjacency = adjacency && !beach[i][e];

    return adjacency;

}

// Allows the user to rent an umbrella that is isolated, i.e. it has not
// any other umbrella rented near it.
// If there is no such umbrella, the function prints an error message
void rent_isolated_umbrella(bool beach[N][M]) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            if (check_adjacent(beach, i, j)) {

                printf("An isolated umbrella has been found\n");
                beach[i][j] = true;
                return;
            }
        }
    }

    printf("Rent wasn't possible. There are no isolated umbrellas\n");
}

// Function for renting adjacent umbrellas.
//
// It gets the row and the number of umbrellas needed.
// If umbrellas are available, they are rented
void rent_adj_umbrella(bool beach[N][M], int row, int n) {
    if (n <= 0 || n > M) {
        printf("Invalid value inserted.\n");

        return;
    }

    bool res = true;
    int count = 0;

    int pos[n];
    memset(pos, 0, sizeof(int) * n);

    for (int j = 0; j < M; j++) {
        if (!beach[row - 1][j] && res && count < n) {
            pos[count] = j;
            count++;
        }
    }

    if (count == n) {
        for (int k = 0; k < n; k++) {
            beach[row - 1][pos[k]] = true;
        }
    }

    printf("Umbrellas correctly rented\n");
}

// Saves beach state to file
void save_state(bool beach[N][M], char *filename) {
    FILE *filePtr = fopen(filename, "w");

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            fprintf(filePtr, " %d", beach[i][j]);
        }

        fprintf(filePtr, "\n");
    }

    fclose(filePtr);
}

// Loads beach state from file
void load_state(bool beach[N][M], char *filename) {
    FILE *filePtr = fopen(filename, "r");

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            fscanf(filePtr, "%d", (int*)&beach[i][j]);
        }
    }

    fclose(filePtr);
}

int main() {
    bool beach[N][M];
    memset(beach, false, M * N);

    const char menu[] =
    	"1. Rent beach umbrella\n"
    	"2. Print state\n"
    	"3. Save state\n"
    	"4. Load state\n"
    	"5. Rent isolated beach umbrella\n"
    	"6. Rent near beach umbrellas\n"
    	"7. Exit\n";

    while(true) {
    	printf("%s\n", menu);

    	int choice;
        printf("Insert your choice: ");
        scanf("%d", &choice);
        printf("\n");

    	switch(choice) {
       	    case 1: {
                int row;
                printf("Insert the row you wish to rent your umbrella: ");
                scanf("%d", &row);
                printf("\n");

                rent_umbrella(beach, row);

           	    break;
            }

           	case 2: {
                printf("Current Beach State:\n");
                print_state(beach);
                printf("\n");

           	    break;
            }

           	case 3: {
                printf("Saving state...\n");

                save_state(beach, "beach.txt");

                printf("State saved!\n");

           	    break;
            }

           	case 4: {
                printf("Loading state...\n");

                load_state(beach, "beach.txt");

                printf("State loaded!\n");

           	    break;
            }

           	case 5: {
                rent_isolated_umbrella(beach);
                printf("\n");

           	    break;
            }

           	case 6: {
                int row, n;

                printf("Insert the row where you want to rent: ");
                scanf("%d", &row);
                printf("\n");

                printf("Insert the number of umbrellas you want to rent: ");
                scanf("%d", &n);
                printf("\n");

                rent_adj_umbrella(beach, row, n);

           	    break;
            }

           	case 7: {
                printf("Goodbye!\n");

           	    return 0;
            }

           	default: {
           	    printf("Invalid choice selected\n");
            }
    	}
    }

    return 1;
}
