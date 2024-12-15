#ifndef DD2375_SHADERINFO_H
#define DD2375_SHADERINFO_H

typedef struct
{
    unsigned int row_dim_x; // Number of rows in X
    unsigned int col_dim_x; // Number of columns in X
    unsigned int inner_dim; // Number of columns in A = number of rows in B
} MatMulParams;

#endif //DD2375_SHADERINFO_H
