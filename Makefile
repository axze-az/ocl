include ../Rules.make
ifeq ($(INSTALLDIR),)
INSTALLDIR=$(HOME)
endif

# Name  of the library
LIBNAME=ocl
MAJOR=0#
MINOR=1#

SLDFLAGS:= $(SLDFLAGS) 
#ARCH=#-march=bdver1 -mxop #-march=bdver1 #-mdispatch-scheduler
ARCH+=#-mpopcnt -mfma
CXXFLAGS+=-I.. -I../stlex -I../thread  -I../sysio -march=native
CXXFLAGS+=-fstrict-aliasing -Wstrict-aliasing=1
CXXFLAGS+=-Wno-error

CSRCS=platform.cc device.cc context.cc buffer.cc queue.cc program.cc

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

TESTPROGS=testocl

tests: $(TESTPROGS)

testocl: testocl.ol
	$(LD) -o $@ $< $(LDFLAGS) -lOpenCL -lstdc++


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
