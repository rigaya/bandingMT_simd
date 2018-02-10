SRCDIR=.
PROGRAM=bandingMT_simd.auf
CXX=g++
LD=gcc
RES=windres
SRCS=bandingMT_simd/banding_sse2.cpp \
     bandingMT_simd/banding_ssse3.cpp \
     bandingMT_simd/banding_sse41.cpp \
     bandingMT_simd/banding_avx.cpp \
     bandingMT_simd/banding_avx2.cpp \
     bandingMT_simd/banding_avx512.cpp \
     bandingMT_simd/banding_func.cpp \
     bandingMT_simd/bandingMT_simd.cpp
RCS=bandingMT_simd/bandingMT_simd.rc
DEF=bandingMT_simd/bandingMT_simd.def

CXXFLAGS=-I./bandingMT_simd --exec-charset=cp932 --input-charset=utf-8 -std=c++11 -m32 -Ofast -fomit-frame-pointer -flto -DNDEBUG=11
LDFLAGS=-flto -static-libgcc -shared -Wl,-s -Wl,--dll,--enable-stdcall-fixup -L. -Wl,-dn,-lstdc++ -lm
RESFLAGS=-c 65001 --input-format=rc --output-format=coff
STRIPFLAGS=

vpath %.cpp $(SRCDIR)
vpath %.rc  $(SRCDIR)

OBJS   = $(SRCS:%.cpp=%.o)
OBJRCS = $(RCS:%.rc=%.rco)

all: $(PROGRAM)

$(PROGRAM): .depend $(OBJS) $(OBJRCS)
	$(LD) $(OBJS) $(OBJRCS) $(DEF) $(LDFLAGS) -o $(PROGRAM)

%.o: %.cpp .depend
	$(eval CXXARCH := '-msse -msse2')
	$(eval CXXARCH := $(shell [ `echo $< | grep 'ssse3'`  ] && echo '$(CXXARCH) -mssse3'          || echo $(CXXARCH)))
	$(eval CXXARCH := $(shell [ `echo $< | grep 'sse41'`  ] && echo '$(CXXARCH) -mssse3 -msse4.1' || echo $(CXXARCH)))
	$(eval CXXARCH := $(shell [ `echo $< | grep 'avx'`    ] && echo '-march=sandybridge'          || echo $(CXXARCH)))
	$(eval CXXARCH := $(shell [ `echo $< | grep 'avx2'`   ] && echo '-march=haswell'              || echo $(CXXARCH)))
	$(eval CXXARCH := $(shell [ `echo $< | grep 'avx512'` ] && echo '-march=skylake-avx512'       || echo $(CXXARCH)))
	$(CXX) -c $(CXXFLAGS) $(CXXARCH) -o $@ $<

%.rco: %.rc
	$(RES) $(RESFLAGS) -o $@ $<
	
.depend:
	@rm -f .depend
	@echo 'generate .depend...'
	@$(foreach SRC, $(SRCS:%=$(SRCDIR)/%), $(CXX) $(SRC) $(CXXFLAGS) -g0 -MT $(SRC:$(SRCDIR)/%.cpp=%.o) -MM >> .depend;)
	
ifneq ($(wildcard .depend),)
include .depend
endif

clean:
	rm -f $(OBJS) $(OBJRCS) $(PROGRAM) .depend
