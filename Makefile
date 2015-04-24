include ../Rules.make
ifeq ($(INSTALLDIR),)
INSTALLDIR=$(HOME)
endif

# Name  of the library
LIBNAME=ocl
MAJOR=0#
MINOR=1#

SLDFLAGS:= $(SLDFLAGS) -lOpenCL
ARCH+=#-mpopcnt -mfma
CXXFLAGS+=-I.. -march=native
CXXFLAGS+=-fstrict-aliasing -Wstrict-aliasing=1
CXXFLAGS+=-Wno-error

CSRCS=impl_devices.cc impl_be_data.cc 

all: lib tests

lib: lib$(LIBNAME).so.$(MAJOR).$(MINOR) lib$(LIBNAME).a lib$(LIBNAME)-g.a

install: install-static install-shared install-debug install-header

###########################################################################
# Installation of the header files
install-header:
	mkdir -p $(INSTALLDIR)/include/cftal
	for i in *.h ; \
	do \
		if [ -f $$i ] ; then \
			$(INSTALL) -c -m 0644 $$i $(INSTALLDIR)/include/cftal	 ; \
		fi ; \
	done

###########################################################################
# Static library
lib$(LIBNAME).a: $(STDOBJS)
	$(AR) $(ARFLAGS) $@ $?

install-static: lib$(LIBNAME).a
	mkdir -p $(INSTALLDIR)/lib
	$(INSTALL) -c -m 0644 lib$(LIBNAME).a $(INSTALLDIR)/lib

###########################################################################
# Static Debug library
lib$(LIBNAME)-g.a: $(DEBOBJS)
	$(AR) $(ARFLAGS) $@ $?

install-debug: lib$(LIBNAME)-g.a
	mkdir -p $(INSTALLDIR)/lib
	$(INSTALL) -c -m 0644 lib$(LIBNAME)-g.a $(INSTALLDIR)/lib


###########################################################################
# Shared library
lib$(LIBNAME).so.$(MAJOR).$(MINOR): $(PICOBJS)
	$(SLD) -Wl,-hlib$(LIBNAME).so.$(MAJOR) \
$(SHAREDLIBS) $(SLDFLAGS) \
-Wl,-Map -Wl,lib$(LIBNAME).map \
-o $@ $(PICOBJS) -lm
	-$(RM) lib$(LIBNAME).so
	-ln -sf lib$(LIBNAME).so.$(MAJOR).$(MINOR) ./lib$(LIBNAME).so.$(MAJOR)
	-ln -sf lib$(LIBNAME).so.$(MAJOR) lib$(LIBNAME).so

install-shared: lib$(LIBNAME).so.$(MAJOR).$(MINOR)
	mkdir -p $(INSTALLDIR)/lib
	$(INSTALL) -m 0755 lib$(LIBNAME).so.$(MAJOR).$(MINOR) $(INSTALLDIR)/lib
	cd $(INSTALLDIR)/lib && \
ln -sf lib$(LIBNAME).so.$(MAJOR).$(MINOR) lib$(LIBNAME).so.$(MAJOR)
	cd $(INSTALLDIR)/lib && \
ln -sf lib$(LIBNAME).so.$(MAJOR) lib$(LIBNAME).so

TESTPROGS=testocl testocl_g test_vector test_vector_g test_be_data

tests: $(TESTPROGS)

testocl: testocl.o
	$(LD) -o $@ $< $(LDFLAGS) -Wl,-rpath=. -L. -locl -lOpenCL -lstdc++ -ldl

testocl_g: testocl.od lib$(LIBNAME)-g.a
	$(LD) -o $@ $< -g $(LDFLAGS) -L. -locl-g -lOpenCL -lstdc++ -ldl

test_be_data: test_be_data.o
	$(LD) -o $@ $< $(LDFLAGS) -Wl,-rpath=. -L. -locl -lOpenCL -lstdc++ -ldl

test_vector: test_vector.o
	$(LD) -o $@ $< $(LDFLAGS) -Wl,-rpath=. -L. -locl -lOpenCL -lstdc++ -ldl

test_vector_g: test_vector.od lib$(LIBNAME)-g.a
	$(LD) -o $@ $< -g $(LDFLAGS) -L. -locl-g -lOpenCL -lstdc++ -ldl


#################################################################
# cleanup
clean:
	-$(RM) -rf *.i *.o* *.so.*  *.a *.so *.map *.s testx86vec
	-$(RM) -rf $(TESTPROGS)

distclean: clean
	-$(RM) -rf .depend .*.dep* *~

#######################################################################
# dependencies
ifneq ($(wildcard ./.*.dep*),)
include $(wildcard ./.*.dep*)
endif
