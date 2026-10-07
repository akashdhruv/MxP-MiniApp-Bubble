# Deforming Bubble mini-app -- build + check
#
# Layout (see dev/tasks/init-project/spec.md, File map):
#   src/*.cpp   -> ./bubble
#   tests/test_*.cpp -> tests/<name> binaries, run by `make check`
#
# No -ffast-math (breaks the 1e-12 reproducibility tolerance, Spec "Silent traps").

CXX      := mpicxx
CXXFLAGS := -std=c++17 -O2 -g -Wall -Wextra -Werror -MMD -MP
LDFLAGS  :=
LDLIBS   := -lm

BIN      := bubble
SRCDIR   := src
OBJDIR   := build
TESTDIR  := tests

SRCS     := $(sort $(wildcard $(SRCDIR)/*.cpp))
OBJS     := $(patsubst $(SRCDIR)/%.cpp,$(OBJDIR)/%.o,$(SRCS))
# everything except main.o, so tests can link the library part
LIBOBJS  := $(filter-out $(OBJDIR)/main.o,$(OBJS))

TESTSRCS := $(sort $(wildcard $(TESTDIR)/test_*.cpp))
TESTBINS := $(patsubst $(TESTDIR)/%.cpp,$(OBJDIR)/%,$(TESTSRCS))

DEPS     := $(OBJS:.o=.d) $(TESTBINS:=.d)

# test_rank_invariance / test_halo need more than one rank
MPIRUN   := mpirun
TIMEOUT  := timeout 600

.PHONY: all clean check debug tests dirs

all: $(BIN)

$(BIN): $(OBJS)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $^ $(LDLIBS)

$(OBJDIR)/%.o: $(SRCDIR)/%.cpp | $(OBJDIR)
	$(CXX) $(CXXFLAGS) -I$(SRCDIR) -c -o $@ $<

$(OBJDIR)/%: $(TESTDIR)/%.cpp $(LIBOBJS) | $(OBJDIR)
	$(CXX) $(CXXFLAGS) -I$(SRCDIR) $(LDFLAGS) -o $@ $< $(LIBOBJS) $(LDLIBS)

$(OBJDIR):
	mkdir -p $(OBJDIR)

tests: $(TESTBINS)

# Each test binary is an MPI program. Ranks it should run on are declared by the
# test itself through a `# ranks: ...` comment on its first lines; default 1.
check: $(BIN) $(TESTBINS)
	@sh tests/run_tests.sh "$(OBJDIR)" $(TESTBINS)

debug:
	$(MAKE) clean
	$(MAKE) CXXFLAGS="-std=c++17 -O1 -g -Wall -Wextra -Werror -MMD -MP -fsanitize=address,undefined" \
	        LDFLAGS="-fsanitize=address,undefined" all tests

clean:
	rm -rf $(OBJDIR) $(BIN)

-include $(DEPS)
