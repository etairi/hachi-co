/* Counts the payload bytes that each MPI rank of PPP26's code sends to the other ranks (scripts/ppp26.py).
 *
 * Loaded with LD_PRELOAD, it wraps the calls of their code that move data through the MPI profiling interface and
 * exports the totals, which scripts/ppp26/timing.patch reads at the boundaries of the timed region. Their own counter
 * misses most messages of the PCS, and an interface counter cannot tell apart ranks that share a machine. A
 * collective counts its logical payload to the other ranks (a root's broadcast to n - 1 ranks, the n - 1 blocks of an
 * all-to-all, a non-root's block of a gather), not the bytes of the algorithm MPI runs for it, which is how our own
 * accounting counts payload bytes too.
 *
 *   mpicc -shared -fPIC -O2 -o libhco_mpi_count.so mpi_count.c
 */
#include <mpi.h>

unsigned long long hco_mpi_sent_bytes = 0;
unsigned long long hco_mpi_sent_msgs = 0;

static unsigned long long payload(int count, MPI_Datatype type) {
    int size = 0;
    PMPI_Type_size(type, &size);
    return (unsigned long long)count * (unsigned long long)size;
}

static int others(MPI_Comm comm) {
    int n = 1;
    PMPI_Comm_size(comm, &n);
    return n - 1;
}

static int rank_of(MPI_Comm comm) {
    int r = 0;
    PMPI_Comm_rank(comm, &r);
    return r;
}

int MPI_Send(const void *buf, int count, MPI_Datatype type, int dest, int tag, MPI_Comm comm) {
    hco_mpi_sent_bytes += payload(count, type);
    hco_mpi_sent_msgs += 1;
    return PMPI_Send(buf, count, type, dest, tag, comm);
}

int MPI_Isend(const void *buf, int count, MPI_Datatype type, int dest, int tag, MPI_Comm comm, MPI_Request *req) {
    hco_mpi_sent_bytes += payload(count, type);
    hco_mpi_sent_msgs += 1;
    return PMPI_Isend(buf, count, type, dest, tag, comm, req);
}

int MPI_Bcast(void *buf, int count, MPI_Datatype type, int root, MPI_Comm comm) {
    if (rank_of(comm) == root) {
        hco_mpi_sent_bytes += payload(count, type) * (unsigned long long)others(comm);
        hco_mpi_sent_msgs += (unsigned long long)others(comm);
    }
    return PMPI_Bcast(buf, count, type, root, comm);
}

int MPI_Alltoall(const void *sbuf, int scount, MPI_Datatype stype, void *rbuf, int rcount, MPI_Datatype rtype,
                 MPI_Comm comm) {
    hco_mpi_sent_bytes += payload(scount, stype) * (unsigned long long)others(comm);
    hco_mpi_sent_msgs += (unsigned long long)others(comm);
    return PMPI_Alltoall(sbuf, scount, stype, rbuf, rcount, rtype, comm);
}

int MPI_Ialltoall(const void *sbuf, int scount, MPI_Datatype stype, void *rbuf, int rcount, MPI_Datatype rtype,
                  MPI_Comm comm, MPI_Request *req) {
    hco_mpi_sent_bytes += payload(scount, stype) * (unsigned long long)others(comm);
    hco_mpi_sent_msgs += (unsigned long long)others(comm);
    return PMPI_Ialltoall(sbuf, scount, stype, rbuf, rcount, rtype, comm, req);
}

int MPI_Gather(const void *sbuf, int scount, MPI_Datatype stype, void *rbuf, int rcount, MPI_Datatype rtype,
               int root, MPI_Comm comm) {
    if (rank_of(comm) != root) {
        hco_mpi_sent_bytes += payload(scount, stype);
        hco_mpi_sent_msgs += 1;
    }
    return PMPI_Gather(sbuf, scount, stype, rbuf, rcount, rtype, root, comm);
}
