??!
??
B
AddV2
x"T
y"T
z"T"
Ttype:
2	??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
?
FloorDiv
x"T
y"T
z"T"
Ttype:
2	
:
FloorMod
x"T
y"T
z"T"
Ttype:
	2	
B
GreaterEqual
x"T
y"T
z
"
Ttype:
2	
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
:
Maximum
x"T
y"T
z"T"
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?
:
Minimum
x"T
y"T
z"T"
Ttype:

2	
=
Mul
x"T
y"T
z"T"
Ttype:
2	?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
>
RealDiv
x"T
y"T
z"T"
Ttype:
2	
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
?
ResourceGather
resource
indices"Tindices
output"dtype"

batch_dimsint "
validate_indicesbool("
dtypetype"
Tindicestype:
2	?
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
-
Sqrt
x"T
y"T"
Ttype:

2
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ?
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
;
Sub
x"T
y"T
z"T"
Ttype:
2	
?
Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.4.02v2.4.0-rc4-71-g582c8d236cb8??
|
dense_45/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??* 
shared_namedense_45/kernel
u
#dense_45/kernel/Read/ReadVariableOpReadVariableOpdense_45/kernel* 
_output_shapes
:
??*
dtype0
s
dense_45/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_45/bias
l
!dense_45/bias/Read/ReadVariableOpReadVariableOpdense_45/bias*
_output_shapes	
:?*
dtype0
|
dense_46/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??* 
shared_namedense_46/kernel
u
#dense_46/kernel/Read/ReadVariableOpReadVariableOpdense_46/kernel* 
_output_shapes
:
??*
dtype0
s
dense_46/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_46/bias
l
!dense_46/bias/Read/ReadVariableOpReadVariableOpdense_46/bias*
_output_shapes	
:?*
dtype0
{
dense_48/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?* 
shared_namedense_48/kernel
t
#dense_48/kernel/Read/ReadVariableOpReadVariableOpdense_48/kernel*
_output_shapes
:	?*
dtype0
s
dense_48/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_48/bias
l
!dense_48/bias/Read/ReadVariableOpReadVariableOpdense_48/bias*
_output_shapes	
:?*
dtype0
|
dense_47/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??* 
shared_namedense_47/kernel
u
#dense_47/kernel/Read/ReadVariableOpReadVariableOpdense_47/kernel* 
_output_shapes
:
??*
dtype0
s
dense_47/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_47/bias
l
!dense_47/bias/Read/ReadVariableOpReadVariableOpdense_47/bias*
_output_shapes	
:?*
dtype0
|
dense_49/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??* 
shared_namedense_49/kernel
u
#dense_49/kernel/Read/ReadVariableOpReadVariableOpdense_49/kernel* 
_output_shapes
:
??*
dtype0
s
dense_49/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_49/bias
l
!dense_49/bias/Read/ReadVariableOpReadVariableOpdense_49/bias*
_output_shapes	
:?*
dtype0
|
dense_50/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??* 
shared_namedense_50/kernel
u
#dense_50/kernel/Read/ReadVariableOpReadVariableOpdense_50/kernel* 
_output_shapes
:
??*
dtype0
s
dense_50/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_50/bias
l
!dense_50/bias/Read/ReadVariableOpReadVariableOpdense_50/bias*
_output_shapes	
:?*
dtype0
|
dense_51/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??* 
shared_namedense_51/kernel
u
#dense_51/kernel/Read/ReadVariableOpReadVariableOpdense_51/kernel* 
_output_shapes
:
??*
dtype0
s
dense_51/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_51/bias
l
!dense_51/bias/Read/ReadVariableOpReadVariableOpdense_51/bias*
_output_shapes	
:?*
dtype0
|
dense_52/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??* 
shared_namedense_52/kernel
u
#dense_52/kernel/Read/ReadVariableOpReadVariableOpdense_52/kernel* 
_output_shapes
:
??*
dtype0
s
dense_52/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_52/bias
l
!dense_52/bias/Read/ReadVariableOpReadVariableOpdense_52/bias*
_output_shapes	
:?*
dtype0
{
dense_53/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?* 
shared_namedense_53/kernel
t
#dense_53/kernel/Read/ReadVariableOpReadVariableOpdense_53/kernel*
_output_shapes
:	?*
dtype0
r
dense_53/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_53/bias
k
!dense_53/bias/Read/ReadVariableOpReadVariableOpdense_53/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
?
embedding_32/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	4?*(
shared_nameembedding_32/embeddings
?
+embedding_32/embeddings/Read/ReadVariableOpReadVariableOpembedding_32/embeddings*
_output_shapes
:	4?*
dtype0
?
embedding_30/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*(
shared_nameembedding_30/embeddings
?
+embedding_30/embeddings/Read/ReadVariableOpReadVariableOpembedding_30/embeddings*
_output_shapes
:	?*
dtype0
?
embedding_31/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*(
shared_nameembedding_31/embeddings
?
+embedding_31/embeddings/Read/ReadVariableOpReadVariableOpembedding_31/embeddings*
_output_shapes
:	?*
dtype0
?
embedding_35/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	4?*(
shared_nameembedding_35/embeddings
?
+embedding_35/embeddings/Read/ReadVariableOpReadVariableOpembedding_35/embeddings*
_output_shapes
:	4?*
dtype0
?
embedding_33/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*(
shared_nameembedding_33/embeddings
?
+embedding_33/embeddings/Read/ReadVariableOpReadVariableOpembedding_33/embeddings*
_output_shapes
:	?*
dtype0
?
embedding_34/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*(
shared_nameembedding_34/embeddings
?
+embedding_34/embeddings/Read/ReadVariableOpReadVariableOpembedding_34/embeddings*
_output_shapes
:	?*
dtype0
?
 normalize_5/normalization_5/meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*1
shared_name" normalize_5/normalization_5/mean
?
4normalize_5/normalization_5/mean/Read/ReadVariableOpReadVariableOp normalize_5/normalization_5/mean*
_output_shapes	
:?*
dtype0
?
$normalize_5/normalization_5/varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*5
shared_name&$normalize_5/normalization_5/variance
?
8normalize_5/normalization_5/variance/Read/ReadVariableOpReadVariableOp$normalize_5/normalization_5/variance*
_output_shapes	
:?*
dtype0
?
!normalize_5/normalization_5/countVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *2
shared_name#!normalize_5/normalization_5/count
?
5normalize_5/normalization_5/count/Read/ReadVariableOpReadVariableOp!normalize_5/normalization_5/count*
_output_shapes
: *
dtype0	
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
?
Adam/dense_45/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*'
shared_nameAdam/dense_45/kernel/m
?
*Adam/dense_45/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_45/kernel/m* 
_output_shapes
:
??*
dtype0
?
Adam/dense_45/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*%
shared_nameAdam/dense_45/bias/m
z
(Adam/dense_45/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_45/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/dense_46/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*'
shared_nameAdam/dense_46/kernel/m
?
*Adam/dense_46/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_46/kernel/m* 
_output_shapes
:
??*
dtype0
?
Adam/dense_46/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*%
shared_nameAdam/dense_46/bias/m
z
(Adam/dense_46/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_46/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/dense_48/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*'
shared_nameAdam/dense_48/kernel/m
?
*Adam/dense_48/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_48/kernel/m*
_output_shapes
:	?*
dtype0
?
Adam/dense_48/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*%
shared_nameAdam/dense_48/bias/m
z
(Adam/dense_48/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_48/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/dense_47/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*'
shared_nameAdam/dense_47/kernel/m
?
*Adam/dense_47/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_47/kernel/m* 
_output_shapes
:
??*
dtype0
?
Adam/dense_47/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*%
shared_nameAdam/dense_47/bias/m
z
(Adam/dense_47/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_47/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/dense_49/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*'
shared_nameAdam/dense_49/kernel/m
?
*Adam/dense_49/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_49/kernel/m* 
_output_shapes
:
??*
dtype0
?
Adam/dense_49/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*%
shared_nameAdam/dense_49/bias/m
z
(Adam/dense_49/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_49/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/dense_50/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*'
shared_nameAdam/dense_50/kernel/m
?
*Adam/dense_50/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_50/kernel/m* 
_output_shapes
:
??*
dtype0
?
Adam/dense_50/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*%
shared_nameAdam/dense_50/bias/m
z
(Adam/dense_50/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_50/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/dense_51/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*'
shared_nameAdam/dense_51/kernel/m
?
*Adam/dense_51/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_51/kernel/m* 
_output_shapes
:
??*
dtype0
?
Adam/dense_51/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*%
shared_nameAdam/dense_51/bias/m
z
(Adam/dense_51/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_51/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/dense_52/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*'
shared_nameAdam/dense_52/kernel/m
?
*Adam/dense_52/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_52/kernel/m* 
_output_shapes
:
??*
dtype0
?
Adam/dense_52/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*%
shared_nameAdam/dense_52/bias/m
z
(Adam/dense_52/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_52/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/dense_53/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*'
shared_nameAdam/dense_53/kernel/m
?
*Adam/dense_53/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_53/kernel/m*
_output_shapes
:	?*
dtype0
?
Adam/dense_53/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_53/bias/m
y
(Adam/dense_53/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_53/bias/m*
_output_shapes
:*
dtype0
?
Adam/embedding_32/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	4?*/
shared_name Adam/embedding_32/embeddings/m
?
2Adam/embedding_32/embeddings/m/Read/ReadVariableOpReadVariableOpAdam/embedding_32/embeddings/m*
_output_shapes
:	4?*
dtype0
?
Adam/embedding_30/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*/
shared_name Adam/embedding_30/embeddings/m
?
2Adam/embedding_30/embeddings/m/Read/ReadVariableOpReadVariableOpAdam/embedding_30/embeddings/m*
_output_shapes
:	?*
dtype0
?
Adam/embedding_31/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*/
shared_name Adam/embedding_31/embeddings/m
?
2Adam/embedding_31/embeddings/m/Read/ReadVariableOpReadVariableOpAdam/embedding_31/embeddings/m*
_output_shapes
:	?*
dtype0
?
Adam/embedding_35/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	4?*/
shared_name Adam/embedding_35/embeddings/m
?
2Adam/embedding_35/embeddings/m/Read/ReadVariableOpReadVariableOpAdam/embedding_35/embeddings/m*
_output_shapes
:	4?*
dtype0
?
Adam/embedding_33/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*/
shared_name Adam/embedding_33/embeddings/m
?
2Adam/embedding_33/embeddings/m/Read/ReadVariableOpReadVariableOpAdam/embedding_33/embeddings/m*
_output_shapes
:	?*
dtype0
?
Adam/embedding_34/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*/
shared_name Adam/embedding_34/embeddings/m
?
2Adam/embedding_34/embeddings/m/Read/ReadVariableOpReadVariableOpAdam/embedding_34/embeddings/m*
_output_shapes
:	?*
dtype0
?
Adam/dense_45/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*'
shared_nameAdam/dense_45/kernel/v
?
*Adam/dense_45/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_45/kernel/v* 
_output_shapes
:
??*
dtype0
?
Adam/dense_45/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*%
shared_nameAdam/dense_45/bias/v
z
(Adam/dense_45/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_45/bias/v*
_output_shapes	
:?*
dtype0
?
Adam/dense_46/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*'
shared_nameAdam/dense_46/kernel/v
?
*Adam/dense_46/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_46/kernel/v* 
_output_shapes
:
??*
dtype0
?
Adam/dense_46/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*%
shared_nameAdam/dense_46/bias/v
z
(Adam/dense_46/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_46/bias/v*
_output_shapes	
:?*
dtype0
?
Adam/dense_48/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*'
shared_nameAdam/dense_48/kernel/v
?
*Adam/dense_48/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_48/kernel/v*
_output_shapes
:	?*
dtype0
?
Adam/dense_48/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*%
shared_nameAdam/dense_48/bias/v
z
(Adam/dense_48/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_48/bias/v*
_output_shapes	
:?*
dtype0
?
Adam/dense_47/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*'
shared_nameAdam/dense_47/kernel/v
?
*Adam/dense_47/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_47/kernel/v* 
_output_shapes
:
??*
dtype0
?
Adam/dense_47/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*%
shared_nameAdam/dense_47/bias/v
z
(Adam/dense_47/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_47/bias/v*
_output_shapes	
:?*
dtype0
?
Adam/dense_49/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*'
shared_nameAdam/dense_49/kernel/v
?
*Adam/dense_49/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_49/kernel/v* 
_output_shapes
:
??*
dtype0
?
Adam/dense_49/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*%
shared_nameAdam/dense_49/bias/v
z
(Adam/dense_49/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_49/bias/v*
_output_shapes	
:?*
dtype0
?
Adam/dense_50/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*'
shared_nameAdam/dense_50/kernel/v
?
*Adam/dense_50/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_50/kernel/v* 
_output_shapes
:
??*
dtype0
?
Adam/dense_50/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*%
shared_nameAdam/dense_50/bias/v
z
(Adam/dense_50/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_50/bias/v*
_output_shapes	
:?*
dtype0
?
Adam/dense_51/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*'
shared_nameAdam/dense_51/kernel/v
?
*Adam/dense_51/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_51/kernel/v* 
_output_shapes
:
??*
dtype0
?
Adam/dense_51/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*%
shared_nameAdam/dense_51/bias/v
z
(Adam/dense_51/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_51/bias/v*
_output_shapes	
:?*
dtype0
?
Adam/dense_52/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*'
shared_nameAdam/dense_52/kernel/v
?
*Adam/dense_52/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_52/kernel/v* 
_output_shapes
:
??*
dtype0
?
Adam/dense_52/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*%
shared_nameAdam/dense_52/bias/v
z
(Adam/dense_52/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_52/bias/v*
_output_shapes	
:?*
dtype0
?
Adam/dense_53/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*'
shared_nameAdam/dense_53/kernel/v
?
*Adam/dense_53/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_53/kernel/v*
_output_shapes
:	?*
dtype0
?
Adam/dense_53/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_53/bias/v
y
(Adam/dense_53/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_53/bias/v*
_output_shapes
:*
dtype0
?
Adam/embedding_32/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	4?*/
shared_name Adam/embedding_32/embeddings/v
?
2Adam/embedding_32/embeddings/v/Read/ReadVariableOpReadVariableOpAdam/embedding_32/embeddings/v*
_output_shapes
:	4?*
dtype0
?
Adam/embedding_30/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*/
shared_name Adam/embedding_30/embeddings/v
?
2Adam/embedding_30/embeddings/v/Read/ReadVariableOpReadVariableOpAdam/embedding_30/embeddings/v*
_output_shapes
:	?*
dtype0
?
Adam/embedding_31/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*/
shared_name Adam/embedding_31/embeddings/v
?
2Adam/embedding_31/embeddings/v/Read/ReadVariableOpReadVariableOpAdam/embedding_31/embeddings/v*
_output_shapes
:	?*
dtype0
?
Adam/embedding_35/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	4?*/
shared_name Adam/embedding_35/embeddings/v
?
2Adam/embedding_35/embeddings/v/Read/ReadVariableOpReadVariableOpAdam/embedding_35/embeddings/v*
_output_shapes
:	4?*
dtype0
?
Adam/embedding_33/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*/
shared_name Adam/embedding_33/embeddings/v
?
2Adam/embedding_33/embeddings/v/Read/ReadVariableOpReadVariableOpAdam/embedding_33/embeddings/v*
_output_shapes
:	?*
dtype0
?
Adam/embedding_34/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*/
shared_name Adam/embedding_34/embeddings/v
?
2Adam/embedding_34/embeddings/v/Read/ReadVariableOpReadVariableOpAdam/embedding_34/embeddings/v*
_output_shapes
:	?*
dtype0
J
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
L
Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    
L
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *    
L
Const_3Const*
_output_shapes
: *
dtype0*
valueB
 * $tI
L
Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *    

NoOpNoOp
??
Const_5Const"/device:CPU:0*
_output_shapes
: *
dtype0*ݘ
valueҘBΘ BƘ
?
layer-0
layer-1
layer-2
layer_with_weights-0
layer-3
layer_with_weights-1
layer-4
layer-5
layer-6
layer-7
	layer-8

layer_with_weights-2

layer-9
layer-10
layer_with_weights-3
layer-11
layer_with_weights-4
layer-12
layer_with_weights-5
layer-13
layer_with_weights-6
layer-14
layer-15
layer_with_weights-7
layer-16
layer-17
layer_with_weights-8
layer-18
layer-19
layer-20
layer_with_weights-9
layer-21
layer-22
layer-23
layer_with_weights-10
layer-24
layer_with_weights-11
layer-25
	optimizer
loss
trainable_variables
	variables
regularization_losses
 	keras_api
!
signatures
 
 
 
?
"layer-0
#layer-1
$layer-2
%layer-3
&layer_with_weights-0
&layer-4
'layer_with_weights-1
'layer-5
(layer-6
)layer-7
*layer-8
+layer_with_weights-2
+layer-9
,layer-10
-layer-11
.layer-12
/layer-13
0layer-14
1trainable_variables
2	variables
3regularization_losses
4	keras_api
?
5layer-0
6layer-1
7layer-2
8layer-3
9layer_with_weights-0
9layer-4
:layer_with_weights-1
:layer-5
;layer-6
<layer-7
=layer-8
>layer_with_weights-2
>layer-9
?layer-10
@layer-11
Alayer-12
Blayer-13
Clayer-14
Dtrainable_variables
E	variables
Fregularization_losses
G	keras_api

H	keras_api

I	keras_api

J	keras_api

K	keras_api
h

Lkernel
Mbias
Ntrainable_variables
O	variables
Pregularization_losses
Q	keras_api

R	keras_api
h

Skernel
Tbias
Utrainable_variables
V	variables
Wregularization_losses
X	keras_api
h

Ykernel
Zbias
[trainable_variables
\	variables
]regularization_losses
^	keras_api
h

_kernel
`bias
atrainable_variables
b	variables
cregularization_losses
d	keras_api
h

ekernel
fbias
gtrainable_variables
h	variables
iregularization_losses
j	keras_api

k	keras_api
h

lkernel
mbias
ntrainable_variables
o	variables
pregularization_losses
q	keras_api

r	keras_api
h

skernel
tbias
utrainable_variables
v	variables
wregularization_losses
x	keras_api

y	keras_api

z	keras_api
i

{kernel
|bias
}trainable_variables
~	variables
regularization_losses
?	keras_api

?	keras_api

?	keras_api
f
?	normalize
?trainable_variables
?	variables
?regularization_losses
?	keras_api
n
?kernel
	?bias
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?
	?iter
?beta_1
?beta_2

?decay
?learning_rateLm?Mm?Sm?Tm?Ym?Zm?_m?`m?em?fm?lm?mm?sm?tm?{m?|m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?Lv?Mv?Sv?Tv?Yv?Zv?_v?`v?ev?fv?lv?mv?sv?tv?{v?|v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?
 
?
?0
?1
?2
?3
?4
?5
L6
M7
S8
T9
Y10
Z11
_12
`13
e14
f15
l16
m17
s18
t19
{20
|21
?22
?23
?
?0
?1
?2
?3
?4
?5
L6
M7
S8
T9
Y10
Z11
_12
`13
e14
f15
l16
m17
s18
t19
{20
|21
?22
?23
?24
?25
?26
 
?
?layers
trainable_variables
 ?layer_regularization_losses
	variables
regularization_losses
?layer_metrics
?metrics
?non_trainable_variables
 
 
V
?trainable_variables
?	variables
?regularization_losses
?	keras_api

?	keras_api

?	keras_api
g
?
embeddings
?trainable_variables
?	variables
?regularization_losses
?	keras_api
g
?
embeddings
?trainable_variables
?	variables
?regularization_losses
?	keras_api

?	keras_api

?	keras_api

?	keras_api
g
?
embeddings
?trainable_variables
?	variables
?regularization_losses
?	keras_api

?	keras_api

?	keras_api

?	keras_api

?	keras_api

?	keras_api

?0
?1
?2

?0
?1
?2
 
?
?layers
1trainable_variables
 ?layer_regularization_losses
2	variables
3regularization_losses
?layer_metrics
?metrics
?non_trainable_variables
 
V
?trainable_variables
?	variables
?regularization_losses
?	keras_api

?	keras_api

?	keras_api
g
?
embeddings
?trainable_variables
?	variables
?regularization_losses
?	keras_api
g
?
embeddings
?trainable_variables
?	variables
?regularization_losses
?	keras_api

?	keras_api

?	keras_api

?	keras_api
g
?
embeddings
?trainable_variables
?	variables
?regularization_losses
?	keras_api

?	keras_api

?	keras_api

?	keras_api

?	keras_api

?	keras_api

?0
?1
?2

?0
?1
?2
 
?
?layers
Dtrainable_variables
 ?layer_regularization_losses
E	variables
Fregularization_losses
?layer_metrics
?metrics
?non_trainable_variables
 
 
 
 
[Y
VARIABLE_VALUEdense_45/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_45/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

L0
M1

L0
M1
 
?
?layers
Ntrainable_variables
 ?layer_regularization_losses
O	variables
Pregularization_losses
?layer_metrics
?metrics
?non_trainable_variables
 
[Y
VARIABLE_VALUEdense_46/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_46/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

S0
T1

S0
T1
 
?
?layers
Utrainable_variables
 ?layer_regularization_losses
V	variables
Wregularization_losses
?layer_metrics
?metrics
?non_trainable_variables
[Y
VARIABLE_VALUEdense_48/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_48/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

Y0
Z1

Y0
Z1
 
?
?layers
[trainable_variables
 ?layer_regularization_losses
\	variables
]regularization_losses
?layer_metrics
?metrics
?non_trainable_variables
[Y
VARIABLE_VALUEdense_47/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_47/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE

_0
`1

_0
`1
 
?
?layers
atrainable_variables
 ?layer_regularization_losses
b	variables
cregularization_losses
?layer_metrics
?metrics
?non_trainable_variables
[Y
VARIABLE_VALUEdense_49/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_49/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE

e0
f1

e0
f1
 
?
?layers
gtrainable_variables
 ?layer_regularization_losses
h	variables
iregularization_losses
?layer_metrics
?metrics
?non_trainable_variables
 
[Y
VARIABLE_VALUEdense_50/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_50/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE

l0
m1

l0
m1
 
?
?layers
ntrainable_variables
 ?layer_regularization_losses
o	variables
pregularization_losses
?layer_metrics
?metrics
?non_trainable_variables
 
[Y
VARIABLE_VALUEdense_51/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_51/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE

s0
t1

s0
t1
 
?
?layers
utrainable_variables
 ?layer_regularization_losses
v	variables
wregularization_losses
?layer_metrics
?metrics
?non_trainable_variables
 
 
[Y
VARIABLE_VALUEdense_52/kernel6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_52/bias4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUE

{0
|1

{0
|1
 
?
?layers
}trainable_variables
 ?layer_regularization_losses
~	variables
regularization_losses
?layer_metrics
?metrics
?non_trainable_variables
 
 
c
?state_variables
?_broadcast_shape
	?mean
?variance

?count
?	keras_api
 

?0
?1
?2
 
?
?layers
?trainable_variables
 ?layer_regularization_losses
?	variables
?regularization_losses
?layer_metrics
?metrics
?non_trainable_variables
\Z
VARIABLE_VALUEdense_53/kernel7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdense_53/bias5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?0
?1
 
?
?layers
?trainable_variables
 ?layer_regularization_losses
?	variables
?regularization_losses
?layer_metrics
?metrics
?non_trainable_variables
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUEembedding_32/embeddings0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUEembedding_30/embeddings0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUEembedding_31/embeddings0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUEembedding_35/embeddings0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUEembedding_33/embeddings0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUEembedding_34/embeddings0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUE normalize_5/normalization_5/mean'variables/22/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUE$normalize_5/normalization_5/variance'variables/23/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUE!normalize_5/normalization_5/count'variables/24/.ATTRIBUTES/VARIABLE_VALUE
?
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
 
 

?0

?0
?1
?2
 
 
 
?
?layers
?trainable_variables
 ?layer_regularization_losses
?	variables
?regularization_losses
?layer_metrics
?metrics
?non_trainable_variables
 
 

?0

?0
 
?
?layers
?trainable_variables
 ?layer_regularization_losses
?	variables
?regularization_losses
?layer_metrics
?metrics
?non_trainable_variables

?0

?0
 
?
?layers
?trainable_variables
 ?layer_regularization_losses
?	variables
?regularization_losses
?layer_metrics
?metrics
?non_trainable_variables
 
 
 

?0

?0
 
?
?layers
?trainable_variables
 ?layer_regularization_losses
?	variables
?regularization_losses
?layer_metrics
?metrics
?non_trainable_variables
 
 
 
 
 
n
"0
#1
$2
%3
&4
'5
(6
)7
*8
+9
,10
-11
.12
/13
014
 
 
 
 
 
 
 
?
?layers
?trainable_variables
 ?layer_regularization_losses
?	variables
?regularization_losses
?layer_metrics
?metrics
?non_trainable_variables
 
 

?0

?0
 
?
?layers
?trainable_variables
 ?layer_regularization_losses
?	variables
?regularization_losses
?layer_metrics
?metrics
?non_trainable_variables

?0

?0
 
?
?layers
?trainable_variables
 ?layer_regularization_losses
?	variables
?regularization_losses
?layer_metrics
?metrics
?non_trainable_variables
 
 
 

?0

?0
 
?
?layers
?trainable_variables
 ?layer_regularization_losses
?	variables
?regularization_losses
?layer_metrics
?metrics
?non_trainable_variables
 
 
 
 
 
n
50
61
72
83
94
:5
;6
<7
=8
>9
?10
@11
A12
B13
C14
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
&
	?mean
?variance

?count
 
 

?0
 
 
 

?0
?1
?2
 
 
 
 
 
8

?total

?count
?	variables
?	keras_api
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?	variables
~|
VARIABLE_VALUEAdam/dense_45/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_45/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_46/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_46/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_48/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_48/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_47/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_47/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_49/kernel/mRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_49/bias/mPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_50/kernel/mRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_50/bias/mPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_51/kernel/mRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_51/bias/mPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_52/kernel/mRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_52/bias/mPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_53/kernel/mSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_53/bias/mQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/embedding_32/embeddings/mLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/embedding_30/embeddings/mLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/embedding_31/embeddings/mLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/embedding_35/embeddings/mLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/embedding_33/embeddings/mLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/embedding_34/embeddings/mLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_45/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_45/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_46/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_46/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_48/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_48/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_47/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_47/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_49/kernel/vRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_49/bias/vPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_50/kernel/vRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_50/bias/vPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_51/kernel/vRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_51/bias/vPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_52/kernel/vRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_52/bias/vPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_53/kernel/vSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_53/bias/vQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/embedding_32/embeddings/vLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/embedding_30/embeddings/vLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/embedding_31/embeddings/vLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/embedding_35/embeddings/vLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/embedding_33/embeddings/vLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/embedding_34/embeddings/vLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
w
serving_default_betsPlaceholder*'
_output_shapes
:?????????
*
dtype0*
shape:?????????

y
serving_default_cards0Placeholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
y
serving_default_cards1Placeholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_betsserving_default_cards0serving_default_cards1ConstConst_1embedding_32/embeddingsembedding_30/embeddingsembedding_31/embeddingsConst_2embedding_35/embeddingsembedding_33/embeddingsembedding_34/embeddingsConst_3Const_4dense_45/kerneldense_45/biasdense_48/kerneldense_48/biasdense_46/kerneldense_46/biasdense_47/kerneldense_47/biasdense_49/kerneldense_49/biasdense_50/kerneldense_50/biasdense_51/kerneldense_51/biasdense_52/kerneldense_52/bias normalize_5/normalization_5/mean$normalize_5/normalization_5/variancedense_53/kerneldense_53/bias*-
Tin&
$2"*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*<
_read_only_resource_inputs
	
 !*2
config_proto" 

CPU

GPU2 *0J 8? */
f*R(
&__inference_signature_wrapper_41026666
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#dense_45/kernel/Read/ReadVariableOp!dense_45/bias/Read/ReadVariableOp#dense_46/kernel/Read/ReadVariableOp!dense_46/bias/Read/ReadVariableOp#dense_48/kernel/Read/ReadVariableOp!dense_48/bias/Read/ReadVariableOp#dense_47/kernel/Read/ReadVariableOp!dense_47/bias/Read/ReadVariableOp#dense_49/kernel/Read/ReadVariableOp!dense_49/bias/Read/ReadVariableOp#dense_50/kernel/Read/ReadVariableOp!dense_50/bias/Read/ReadVariableOp#dense_51/kernel/Read/ReadVariableOp!dense_51/bias/Read/ReadVariableOp#dense_52/kernel/Read/ReadVariableOp!dense_52/bias/Read/ReadVariableOp#dense_53/kernel/Read/ReadVariableOp!dense_53/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp+embedding_32/embeddings/Read/ReadVariableOp+embedding_30/embeddings/Read/ReadVariableOp+embedding_31/embeddings/Read/ReadVariableOp+embedding_35/embeddings/Read/ReadVariableOp+embedding_33/embeddings/Read/ReadVariableOp+embedding_34/embeddings/Read/ReadVariableOp4normalize_5/normalization_5/mean/Read/ReadVariableOp8normalize_5/normalization_5/variance/Read/ReadVariableOp5normalize_5/normalization_5/count/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp*Adam/dense_45/kernel/m/Read/ReadVariableOp(Adam/dense_45/bias/m/Read/ReadVariableOp*Adam/dense_46/kernel/m/Read/ReadVariableOp(Adam/dense_46/bias/m/Read/ReadVariableOp*Adam/dense_48/kernel/m/Read/ReadVariableOp(Adam/dense_48/bias/m/Read/ReadVariableOp*Adam/dense_47/kernel/m/Read/ReadVariableOp(Adam/dense_47/bias/m/Read/ReadVariableOp*Adam/dense_49/kernel/m/Read/ReadVariableOp(Adam/dense_49/bias/m/Read/ReadVariableOp*Adam/dense_50/kernel/m/Read/ReadVariableOp(Adam/dense_50/bias/m/Read/ReadVariableOp*Adam/dense_51/kernel/m/Read/ReadVariableOp(Adam/dense_51/bias/m/Read/ReadVariableOp*Adam/dense_52/kernel/m/Read/ReadVariableOp(Adam/dense_52/bias/m/Read/ReadVariableOp*Adam/dense_53/kernel/m/Read/ReadVariableOp(Adam/dense_53/bias/m/Read/ReadVariableOp2Adam/embedding_32/embeddings/m/Read/ReadVariableOp2Adam/embedding_30/embeddings/m/Read/ReadVariableOp2Adam/embedding_31/embeddings/m/Read/ReadVariableOp2Adam/embedding_35/embeddings/m/Read/ReadVariableOp2Adam/embedding_33/embeddings/m/Read/ReadVariableOp2Adam/embedding_34/embeddings/m/Read/ReadVariableOp*Adam/dense_45/kernel/v/Read/ReadVariableOp(Adam/dense_45/bias/v/Read/ReadVariableOp*Adam/dense_46/kernel/v/Read/ReadVariableOp(Adam/dense_46/bias/v/Read/ReadVariableOp*Adam/dense_48/kernel/v/Read/ReadVariableOp(Adam/dense_48/bias/v/Read/ReadVariableOp*Adam/dense_47/kernel/v/Read/ReadVariableOp(Adam/dense_47/bias/v/Read/ReadVariableOp*Adam/dense_49/kernel/v/Read/ReadVariableOp(Adam/dense_49/bias/v/Read/ReadVariableOp*Adam/dense_50/kernel/v/Read/ReadVariableOp(Adam/dense_50/bias/v/Read/ReadVariableOp*Adam/dense_51/kernel/v/Read/ReadVariableOp(Adam/dense_51/bias/v/Read/ReadVariableOp*Adam/dense_52/kernel/v/Read/ReadVariableOp(Adam/dense_52/bias/v/Read/ReadVariableOp*Adam/dense_53/kernel/v/Read/ReadVariableOp(Adam/dense_53/bias/v/Read/ReadVariableOp2Adam/embedding_32/embeddings/v/Read/ReadVariableOp2Adam/embedding_30/embeddings/v/Read/ReadVariableOp2Adam/embedding_31/embeddings/v/Read/ReadVariableOp2Adam/embedding_35/embeddings/v/Read/ReadVariableOp2Adam/embedding_33/embeddings/v/Read/ReadVariableOp2Adam/embedding_34/embeddings/v/Read/ReadVariableOpConst_5*_
TinX
V2T		*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? **
f%R#
!__inference__traced_save_41027964
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_45/kerneldense_45/biasdense_46/kerneldense_46/biasdense_48/kerneldense_48/biasdense_47/kerneldense_47/biasdense_49/kerneldense_49/biasdense_50/kerneldense_50/biasdense_51/kerneldense_51/biasdense_52/kerneldense_52/biasdense_53/kerneldense_53/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_rateembedding_32/embeddingsembedding_30/embeddingsembedding_31/embeddingsembedding_35/embeddingsembedding_33/embeddingsembedding_34/embeddings normalize_5/normalization_5/mean$normalize_5/normalization_5/variance!normalize_5/normalization_5/counttotalcountAdam/dense_45/kernel/mAdam/dense_45/bias/mAdam/dense_46/kernel/mAdam/dense_46/bias/mAdam/dense_48/kernel/mAdam/dense_48/bias/mAdam/dense_47/kernel/mAdam/dense_47/bias/mAdam/dense_49/kernel/mAdam/dense_49/bias/mAdam/dense_50/kernel/mAdam/dense_50/bias/mAdam/dense_51/kernel/mAdam/dense_51/bias/mAdam/dense_52/kernel/mAdam/dense_52/bias/mAdam/dense_53/kernel/mAdam/dense_53/bias/mAdam/embedding_32/embeddings/mAdam/embedding_30/embeddings/mAdam/embedding_31/embeddings/mAdam/embedding_35/embeddings/mAdam/embedding_33/embeddings/mAdam/embedding_34/embeddings/mAdam/dense_45/kernel/vAdam/dense_45/bias/vAdam/dense_46/kernel/vAdam/dense_46/bias/vAdam/dense_48/kernel/vAdam/dense_48/bias/vAdam/dense_47/kernel/vAdam/dense_47/bias/vAdam/dense_49/kernel/vAdam/dense_49/bias/vAdam/dense_50/kernel/vAdam/dense_50/bias/vAdam/dense_51/kernel/vAdam/dense_51/bias/vAdam/dense_52/kernel/vAdam/dense_52/bias/vAdam/dense_53/kernel/vAdam/dense_53/bias/vAdam/embedding_32/embeddings/vAdam/embedding_30/embeddings/vAdam/embedding_31/embeddings/vAdam/embedding_35/embeddings/vAdam/embedding_33/embeddings/vAdam/embedding_34/embeddings/v*^
TinW
U2S*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *-
f(R&
$__inference__traced_restore_41028220??
?
?
+__inference_dense_46_layer_call_fn_41027404

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_dense_46_layer_call_and_return_conditional_losses_410259582
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
F__inference_dense_47_layer_call_and_return_conditional_losses_41027434

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
u
/__inference_embedding_35_layer_call_fn_41027654

inputs
unknown
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *S
fNRL
J__inference_embedding_35_layer_call_and_return_conditional_losses_410256192
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????:22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
u
/__inference_embedding_33_layer_call_fn_41027671

inputs
unknown
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *S
fNRL
J__inference_embedding_33_layer_call_and_return_conditional_losses_410256412
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????:22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
1__inference_custom_model_5_layer_call_fn_41027075

inputs_0_0

inputs_0_1
inputs_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCall
inputs_0_0
inputs_0_1inputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29*-
Tin&
$2"*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*<
_read_only_resource_inputs
	
 !*2
config_proto" 

CPU

GPU2 *0J 8? *U
fPRN
L__inference_custom_model_5_layer_call_and_return_conditional_losses_410263612
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????:?????????:?????????
: : :::: :::: : ::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
'
_output_shapes
:?????????
$
_user_specified_name
inputs/0/0:SO
'
_output_shapes
:?????????
$
_user_specified_name
inputs/0/1:QM
'
_output_shapes
:?????????

"
_user_specified_name
inputs/1:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?Y
?

L__inference_custom_model_5_layer_call_and_return_conditional_losses_41026361

inputs
inputs_1
inputs_2+
'tf_math_greater_equal_17_greaterequal_y
model_10_41026276
model_10_41026278
model_10_41026280
model_10_41026282
model_11_41026285
model_11_41026287
model_11_41026289
model_11_41026291/
+tf_clip_by_value_17_clip_by_value_minimum_y'
#tf_clip_by_value_17_clip_by_value_y
dense_45_41026303
dense_45_41026305
dense_48_41026308
dense_48_41026310
dense_46_41026313
dense_46_41026315
dense_47_41026318
dense_47_41026320
dense_49_41026323
dense_49_41026325
dense_50_41026330
dense_50_41026332
dense_51_41026336
dense_51_41026338
dense_52_41026343
dense_52_41026345
normalize_5_41026350
normalize_5_41026352
dense_53_41026355
dense_53_41026357
identity?? dense_45/StatefulPartitionedCall? dense_46/StatefulPartitionedCall? dense_47/StatefulPartitionedCall? dense_48/StatefulPartitionedCall? dense_49/StatefulPartitionedCall? dense_50/StatefulPartitionedCall? dense_51/StatefulPartitionedCall? dense_52/StatefulPartitionedCall? dense_53/StatefulPartitionedCall? model_10/StatefulPartitionedCall? model_11/StatefulPartitionedCall?#normalize_5/StatefulPartitionedCall?
%tf.math.greater_equal_17/GreaterEqualGreaterEqualinputs_2'tf_math_greater_equal_17_greaterequal_y*
T0*'
_output_shapes
:?????????
2'
%tf.math.greater_equal_17/GreaterEqual?
 model_10/StatefulPartitionedCallStatefulPartitionedCallinputsmodel_10_41026276model_10_41026278model_10_41026280model_10_41026282*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*%
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_model_10_layer_call_and_return_conditional_losses_410255252"
 model_10/StatefulPartitionedCall?
 model_11/StatefulPartitionedCallStatefulPartitionedCallinputs_1model_11_41026285model_11_41026287model_11_41026289model_11_41026291*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*%
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_model_11_layer_call_and_return_conditional_losses_410257512"
 model_11/StatefulPartitionedCall?
)tf.clip_by_value_17/clip_by_value/MinimumMinimuminputs_2+tf_clip_by_value_17_clip_by_value_minimum_y*
T0*'
_output_shapes
:?????????
2+
)tf.clip_by_value_17/clip_by_value/Minimum?
!tf.clip_by_value_17/clip_by_valueMaximum-tf.clip_by_value_17/clip_by_value/Minimum:z:0#tf_clip_by_value_17_clip_by_value_y*
T0*'
_output_shapes
:?????????
2#
!tf.clip_by_value_17/clip_by_value?
tf.cast_17/CastCast)tf.math.greater_equal_17/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????
2
tf.cast_17/Castv
tf.concat_15/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
tf.concat_15/concat/axis?
tf.concat_15/concatConcatV2)model_10/StatefulPartitionedCall:output:0)model_11/StatefulPartitionedCall:output:0!tf.concat_15/concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????2
tf.concat_15/concat
tf.concat_16/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
tf.concat_16/concat/axis?
tf.concat_16/concatConcatV2%tf.clip_by_value_17/clip_by_value:z:0tf.cast_17/Cast:y:0!tf.concat_16/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????2
tf.concat_16/concat?
 dense_45/StatefulPartitionedCallStatefulPartitionedCalltf.concat_15/concat:output:0dense_45_41026303dense_45_41026305*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_dense_45_layer_call_and_return_conditional_losses_410259052"
 dense_45/StatefulPartitionedCall?
 dense_48/StatefulPartitionedCallStatefulPartitionedCalltf.concat_16/concat:output:0dense_48_41026308dense_48_41026310*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_dense_48_layer_call_and_return_conditional_losses_410259312"
 dense_48/StatefulPartitionedCall?
 dense_46/StatefulPartitionedCallStatefulPartitionedCall)dense_45/StatefulPartitionedCall:output:0dense_46_41026313dense_46_41026315*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_dense_46_layer_call_and_return_conditional_losses_410259582"
 dense_46/StatefulPartitionedCall?
 dense_47/StatefulPartitionedCallStatefulPartitionedCall)dense_46/StatefulPartitionedCall:output:0dense_47_41026318dense_47_41026320*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_dense_47_layer_call_and_return_conditional_losses_410259852"
 dense_47/StatefulPartitionedCall?
 dense_49/StatefulPartitionedCallStatefulPartitionedCall)dense_48/StatefulPartitionedCall:output:0dense_49_41026323dense_49_41026325*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_dense_49_layer_call_and_return_conditional_losses_410260112"
 dense_49/StatefulPartitionedCall
tf.concat_17/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
tf.concat_17/concat/axis?
tf.concat_17/concatConcatV2)dense_47/StatefulPartitionedCall:output:0)dense_49/StatefulPartitionedCall:output:0!tf.concat_17/concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????2
tf.concat_17/concat?
 dense_50/StatefulPartitionedCallStatefulPartitionedCalltf.concat_17/concat:output:0dense_50_41026330dense_50_41026332*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_dense_50_layer_call_and_return_conditional_losses_410260392"
 dense_50/StatefulPartitionedCall?
tf.nn.relu_15/ReluRelu)dense_50/StatefulPartitionedCall:output:0*
T0*(
_output_shapes
:??????????2
tf.nn.relu_15/Relu?
 dense_51/StatefulPartitionedCallStatefulPartitionedCall tf.nn.relu_15/Relu:activations:0dense_51_41026336dense_51_41026338*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_dense_51_layer_call_and_return_conditional_losses_410260662"
 dense_51/StatefulPartitionedCall?
tf.__operators__.add_34/AddV2AddV2)dense_51/StatefulPartitionedCall:output:0 tf.nn.relu_15/Relu:activations:0*
T0*(
_output_shapes
:??????????2
tf.__operators__.add_34/AddV2?
tf.nn.relu_16/ReluRelu!tf.__operators__.add_34/AddV2:z:0*
T0*(
_output_shapes
:??????????2
tf.nn.relu_16/Relu?
 dense_52/StatefulPartitionedCallStatefulPartitionedCall tf.nn.relu_16/Relu:activations:0dense_52_41026343dense_52_41026345*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_dense_52_layer_call_and_return_conditional_losses_410260942"
 dense_52/StatefulPartitionedCall?
tf.__operators__.add_35/AddV2AddV2)dense_52/StatefulPartitionedCall:output:0 tf.nn.relu_16/Relu:activations:0*
T0*(
_output_shapes
:??????????2
tf.__operators__.add_35/AddV2?
tf.nn.relu_17/ReluRelu!tf.__operators__.add_35/AddV2:z:0*
T0*(
_output_shapes
:??????????2
tf.nn.relu_17/Relu?
#normalize_5/StatefulPartitionedCallStatefulPartitionedCall tf.nn.relu_17/Relu:activations:0normalize_5_41026350normalize_5_41026352*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *R
fMRK
I__inference_normalize_5_layer_call_and_return_conditional_losses_410261292%
#normalize_5/StatefulPartitionedCall?
 dense_53/StatefulPartitionedCallStatefulPartitionedCall,normalize_5/StatefulPartitionedCall:output:0dense_53_41026355dense_53_41026357*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_dense_53_layer_call_and_return_conditional_losses_410261552"
 dense_53/StatefulPartitionedCall?
IdentityIdentity)dense_53/StatefulPartitionedCall:output:0!^dense_45/StatefulPartitionedCall!^dense_46/StatefulPartitionedCall!^dense_47/StatefulPartitionedCall!^dense_48/StatefulPartitionedCall!^dense_49/StatefulPartitionedCall!^dense_50/StatefulPartitionedCall!^dense_51/StatefulPartitionedCall!^dense_52/StatefulPartitionedCall!^dense_53/StatefulPartitionedCall!^model_10/StatefulPartitionedCall!^model_11/StatefulPartitionedCall$^normalize_5/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????:?????????:?????????
: : :::: :::: : ::::::::::::::::::::2D
 dense_45/StatefulPartitionedCall dense_45/StatefulPartitionedCall2D
 dense_46/StatefulPartitionedCall dense_46/StatefulPartitionedCall2D
 dense_47/StatefulPartitionedCall dense_47/StatefulPartitionedCall2D
 dense_48/StatefulPartitionedCall dense_48/StatefulPartitionedCall2D
 dense_49/StatefulPartitionedCall dense_49/StatefulPartitionedCall2D
 dense_50/StatefulPartitionedCall dense_50/StatefulPartitionedCall2D
 dense_51/StatefulPartitionedCall dense_51/StatefulPartitionedCall2D
 dense_52/StatefulPartitionedCall dense_52/StatefulPartitionedCall2D
 dense_53/StatefulPartitionedCall dense_53/StatefulPartitionedCall2D
 model_10/StatefulPartitionedCall model_10/StatefulPartitionedCall2D
 model_11/StatefulPartitionedCall model_11/StatefulPartitionedCall2J
#normalize_5/StatefulPartitionedCall#normalize_5/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????

 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
+__inference_model_10_layer_call_fn_41027241

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*%
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_model_10_layer_call_and_return_conditional_losses_410255252
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:?????????: :::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: 
?.
?
F__inference_model_11_layer_call_and_return_conditional_losses_41025716
input_12+
'tf_math_greater_equal_16_greaterequal_y
embedding_35_41025698
embedding_33_41025701
embedding_34_41025706
identity??$embedding_33/StatefulPartitionedCall?$embedding_34/StatefulPartitionedCall?$embedding_35/StatefulPartitionedCall?
flatten_11/PartitionedCallPartitionedCallinput_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *Q
fLRJ
H__inference_flatten_11_layer_call_and_return_conditional_losses_410255912
flatten_11/PartitionedCall?
+tf.clip_by_value_16/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 * $tI2-
+tf.clip_by_value_16/clip_by_value/Minimum/y?
)tf.clip_by_value_16/clip_by_value/MinimumMinimum#flatten_11/PartitionedCall:output:04tf.clip_by_value_16/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2+
)tf.clip_by_value_16/clip_by_value/Minimum?
#tf.clip_by_value_16/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2%
#tf.clip_by_value_16/clip_by_value/y?
!tf.clip_by_value_16/clip_by_valueMaximum-tf.clip_by_value_16/clip_by_value/Minimum:z:0,tf.clip_by_value_16/clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????2#
!tf.clip_by_value_16/clip_by_value?
$tf.compat.v1.floor_div_11/FloorDiv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2&
$tf.compat.v1.floor_div_11/FloorDiv/y?
"tf.compat.v1.floor_div_11/FloorDivFloorDiv%tf.clip_by_value_16/clip_by_value:z:0-tf.compat.v1.floor_div_11/FloorDiv/y:output:0*
T0*'
_output_shapes
:?????????2$
"tf.compat.v1.floor_div_11/FloorDiv?
%tf.math.greater_equal_16/GreaterEqualGreaterEqual#flatten_11/PartitionedCall:output:0'tf_math_greater_equal_16_greaterequal_y*
T0*'
_output_shapes
:?????????2'
%tf.math.greater_equal_16/GreaterEqual?
tf.math.floormod_11/FloorMod/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2 
tf.math.floormod_11/FloorMod/y?
tf.math.floormod_11/FloorModFloorMod%tf.clip_by_value_16/clip_by_value:z:0'tf.math.floormod_11/FloorMod/y:output:0*
T0*'
_output_shapes
:?????????2
tf.math.floormod_11/FloorMod?
$embedding_35/StatefulPartitionedCallStatefulPartitionedCall%tf.clip_by_value_16/clip_by_value:z:0embedding_35_41025698*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *S
fNRL
J__inference_embedding_35_layer_call_and_return_conditional_losses_410256192&
$embedding_35/StatefulPartitionedCall?
$embedding_33/StatefulPartitionedCallStatefulPartitionedCall&tf.compat.v1.floor_div_11/FloorDiv:z:0embedding_33_41025701*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *S
fNRL
J__inference_embedding_33_layer_call_and_return_conditional_losses_410256412&
$embedding_33/StatefulPartitionedCall?
tf.cast_16/CastCast)tf.math.greater_equal_16/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2
tf.cast_16/Cast?
tf.__operators__.add_32/AddV2AddV2-embedding_35/StatefulPartitionedCall:output:0-embedding_33/StatefulPartitionedCall:output:0*
T0*,
_output_shapes
:??????????2
tf.__operators__.add_32/AddV2?
$embedding_34/StatefulPartitionedCallStatefulPartitionedCall tf.math.floormod_11/FloorMod:z:0embedding_34_41025706*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *S
fNRL
J__inference_embedding_34_layer_call_and_return_conditional_losses_410256652&
$embedding_34/StatefulPartitionedCall?
tf.__operators__.add_33/AddV2AddV2!tf.__operators__.add_32/AddV2:z:0-embedding_34/StatefulPartitionedCall:output:0*
T0*,
_output_shapes
:??????????2
tf.__operators__.add_33/AddV2?
 tf.expand_dims_11/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2"
 tf.expand_dims_11/ExpandDims/dim?
tf.expand_dims_11/ExpandDims
ExpandDimstf.cast_16/Cast:y:0)tf.expand_dims_11/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????2
tf.expand_dims_11/ExpandDims?
tf.math.multiply_11/MulMul!tf.__operators__.add_33/AddV2:z:0%tf.expand_dims_11/ExpandDims:output:0*
T0*,
_output_shapes
:??????????2
tf.math.multiply_11/Mul?
+tf.math.reduce_sum_11/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2-
+tf.math.reduce_sum_11/Sum/reduction_indices?
tf.math.reduce_sum_11/SumSumtf.math.multiply_11/Mul:z:04tf.math.reduce_sum_11/Sum/reduction_indices:output:0*
T0*(
_output_shapes
:??????????2
tf.math.reduce_sum_11/Sum?
IdentityIdentity"tf.math.reduce_sum_11/Sum:output:0%^embedding_33/StatefulPartitionedCall%^embedding_34/StatefulPartitionedCall%^embedding_35/StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:?????????: :::2L
$embedding_33/StatefulPartitionedCall$embedding_33/StatefulPartitionedCall2L
$embedding_34/StatefulPartitionedCall$embedding_34/StatefulPartitionedCall2L
$embedding_35/StatefulPartitionedCall$embedding_35/StatefulPartitionedCall:Q M
'
_output_shapes
:?????????
"
_user_specified_name
input_12:

_output_shapes
: 
?9
?
F__inference_model_11_layer_call_and_return_conditional_losses_41027296

inputs+
'tf_math_greater_equal_16_greaterequal_y*
&embedding_35_embedding_lookup_41027270*
&embedding_33_embedding_lookup_41027276*
&embedding_34_embedding_lookup_41027284
identity??embedding_33/embedding_lookup?embedding_34/embedding_lookup?embedding_35/embedding_lookupu
flatten_11/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
flatten_11/Const?
flatten_11/ReshapeReshapeinputsflatten_11/Const:output:0*
T0*'
_output_shapes
:?????????2
flatten_11/Reshape?
+tf.clip_by_value_16/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 * $tI2-
+tf.clip_by_value_16/clip_by_value/Minimum/y?
)tf.clip_by_value_16/clip_by_value/MinimumMinimumflatten_11/Reshape:output:04tf.clip_by_value_16/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2+
)tf.clip_by_value_16/clip_by_value/Minimum?
#tf.clip_by_value_16/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2%
#tf.clip_by_value_16/clip_by_value/y?
!tf.clip_by_value_16/clip_by_valueMaximum-tf.clip_by_value_16/clip_by_value/Minimum:z:0,tf.clip_by_value_16/clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????2#
!tf.clip_by_value_16/clip_by_value?
$tf.compat.v1.floor_div_11/FloorDiv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2&
$tf.compat.v1.floor_div_11/FloorDiv/y?
"tf.compat.v1.floor_div_11/FloorDivFloorDiv%tf.clip_by_value_16/clip_by_value:z:0-tf.compat.v1.floor_div_11/FloorDiv/y:output:0*
T0*'
_output_shapes
:?????????2$
"tf.compat.v1.floor_div_11/FloorDiv?
%tf.math.greater_equal_16/GreaterEqualGreaterEqualflatten_11/Reshape:output:0'tf_math_greater_equal_16_greaterequal_y*
T0*'
_output_shapes
:?????????2'
%tf.math.greater_equal_16/GreaterEqual?
tf.math.floormod_11/FloorMod/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2 
tf.math.floormod_11/FloorMod/y?
tf.math.floormod_11/FloorModFloorMod%tf.clip_by_value_16/clip_by_value:z:0'tf.math.floormod_11/FloorMod/y:output:0*
T0*'
_output_shapes
:?????????2
tf.math.floormod_11/FloorMod?
embedding_35/CastCast%tf.clip_by_value_16/clip_by_value:z:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2
embedding_35/Cast?
embedding_35/embedding_lookupResourceGather&embedding_35_embedding_lookup_41027270embedding_35/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*9
_class/
-+loc:@embedding_35/embedding_lookup/41027270*,
_output_shapes
:??????????*
dtype02
embedding_35/embedding_lookup?
&embedding_35/embedding_lookup/IdentityIdentity&embedding_35/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*9
_class/
-+loc:@embedding_35/embedding_lookup/41027270*,
_output_shapes
:??????????2(
&embedding_35/embedding_lookup/Identity?
(embedding_35/embedding_lookup/Identity_1Identity/embedding_35/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????2*
(embedding_35/embedding_lookup/Identity_1?
embedding_33/CastCast&tf.compat.v1.floor_div_11/FloorDiv:z:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2
embedding_33/Cast?
embedding_33/embedding_lookupResourceGather&embedding_33_embedding_lookup_41027276embedding_33/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*9
_class/
-+loc:@embedding_33/embedding_lookup/41027276*,
_output_shapes
:??????????*
dtype02
embedding_33/embedding_lookup?
&embedding_33/embedding_lookup/IdentityIdentity&embedding_33/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*9
_class/
-+loc:@embedding_33/embedding_lookup/41027276*,
_output_shapes
:??????????2(
&embedding_33/embedding_lookup/Identity?
(embedding_33/embedding_lookup/Identity_1Identity/embedding_33/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????2*
(embedding_33/embedding_lookup/Identity_1?
tf.cast_16/CastCast)tf.math.greater_equal_16/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2
tf.cast_16/Cast?
tf.__operators__.add_32/AddV2AddV21embedding_35/embedding_lookup/Identity_1:output:01embedding_33/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:??????????2
tf.__operators__.add_32/AddV2?
embedding_34/CastCast tf.math.floormod_11/FloorMod:z:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2
embedding_34/Cast?
embedding_34/embedding_lookupResourceGather&embedding_34_embedding_lookup_41027284embedding_34/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*9
_class/
-+loc:@embedding_34/embedding_lookup/41027284*,
_output_shapes
:??????????*
dtype02
embedding_34/embedding_lookup?
&embedding_34/embedding_lookup/IdentityIdentity&embedding_34/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*9
_class/
-+loc:@embedding_34/embedding_lookup/41027284*,
_output_shapes
:??????????2(
&embedding_34/embedding_lookup/Identity?
(embedding_34/embedding_lookup/Identity_1Identity/embedding_34/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????2*
(embedding_34/embedding_lookup/Identity_1?
tf.__operators__.add_33/AddV2AddV2!tf.__operators__.add_32/AddV2:z:01embedding_34/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:??????????2
tf.__operators__.add_33/AddV2?
 tf.expand_dims_11/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2"
 tf.expand_dims_11/ExpandDims/dim?
tf.expand_dims_11/ExpandDims
ExpandDimstf.cast_16/Cast:y:0)tf.expand_dims_11/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????2
tf.expand_dims_11/ExpandDims?
tf.math.multiply_11/MulMul!tf.__operators__.add_33/AddV2:z:0%tf.expand_dims_11/ExpandDims:output:0*
T0*,
_output_shapes
:??????????2
tf.math.multiply_11/Mul?
+tf.math.reduce_sum_11/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2-
+tf.math.reduce_sum_11/Sum/reduction_indices?
tf.math.reduce_sum_11/SumSumtf.math.multiply_11/Mul:z:04tf.math.reduce_sum_11/Sum/reduction_indices:output:0*
T0*(
_output_shapes
:??????????2
tf.math.reduce_sum_11/Sum?
IdentityIdentity"tf.math.reduce_sum_11/Sum:output:0^embedding_33/embedding_lookup^embedding_34/embedding_lookup^embedding_35/embedding_lookup*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:?????????: :::2>
embedding_33/embedding_lookupembedding_33/embedding_lookup2>
embedding_34/embedding_lookupembedding_34/embedding_lookup2>
embedding_35/embedding_lookupembedding_35/embedding_lookup:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: 
?.
?
F__inference_model_10_layer_call_and_return_conditional_losses_41025458
input_11+
'tf_math_greater_equal_15_greaterequal_y
embedding_32_41025402
embedding_30_41025424
embedding_31_41025448
identity??$embedding_30/StatefulPartitionedCall?$embedding_31/StatefulPartitionedCall?$embedding_32/StatefulPartitionedCall?
flatten_10/PartitionedCallPartitionedCallinput_11*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *Q
fLRJ
H__inference_flatten_10_layer_call_and_return_conditional_losses_410253652
flatten_10/PartitionedCall?
+tf.clip_by_value_15/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 * $tI2-
+tf.clip_by_value_15/clip_by_value/Minimum/y?
)tf.clip_by_value_15/clip_by_value/MinimumMinimum#flatten_10/PartitionedCall:output:04tf.clip_by_value_15/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2+
)tf.clip_by_value_15/clip_by_value/Minimum?
#tf.clip_by_value_15/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2%
#tf.clip_by_value_15/clip_by_value/y?
!tf.clip_by_value_15/clip_by_valueMaximum-tf.clip_by_value_15/clip_by_value/Minimum:z:0,tf.clip_by_value_15/clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????2#
!tf.clip_by_value_15/clip_by_value?
$tf.compat.v1.floor_div_10/FloorDiv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2&
$tf.compat.v1.floor_div_10/FloorDiv/y?
"tf.compat.v1.floor_div_10/FloorDivFloorDiv%tf.clip_by_value_15/clip_by_value:z:0-tf.compat.v1.floor_div_10/FloorDiv/y:output:0*
T0*'
_output_shapes
:?????????2$
"tf.compat.v1.floor_div_10/FloorDiv?
%tf.math.greater_equal_15/GreaterEqualGreaterEqual#flatten_10/PartitionedCall:output:0'tf_math_greater_equal_15_greaterequal_y*
T0*'
_output_shapes
:?????????2'
%tf.math.greater_equal_15/GreaterEqual?
tf.math.floormod_10/FloorMod/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2 
tf.math.floormod_10/FloorMod/y?
tf.math.floormod_10/FloorModFloorMod%tf.clip_by_value_15/clip_by_value:z:0'tf.math.floormod_10/FloorMod/y:output:0*
T0*'
_output_shapes
:?????????2
tf.math.floormod_10/FloorMod?
$embedding_32/StatefulPartitionedCallStatefulPartitionedCall%tf.clip_by_value_15/clip_by_value:z:0embedding_32_41025402*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *S
fNRL
J__inference_embedding_32_layer_call_and_return_conditional_losses_410253932&
$embedding_32/StatefulPartitionedCall?
$embedding_30/StatefulPartitionedCallStatefulPartitionedCall&tf.compat.v1.floor_div_10/FloorDiv:z:0embedding_30_41025424*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *S
fNRL
J__inference_embedding_30_layer_call_and_return_conditional_losses_410254152&
$embedding_30/StatefulPartitionedCall?
tf.cast_15/CastCast)tf.math.greater_equal_15/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2
tf.cast_15/Cast?
tf.__operators__.add_30/AddV2AddV2-embedding_32/StatefulPartitionedCall:output:0-embedding_30/StatefulPartitionedCall:output:0*
T0*,
_output_shapes
:??????????2
tf.__operators__.add_30/AddV2?
$embedding_31/StatefulPartitionedCallStatefulPartitionedCall tf.math.floormod_10/FloorMod:z:0embedding_31_41025448*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *S
fNRL
J__inference_embedding_31_layer_call_and_return_conditional_losses_410254392&
$embedding_31/StatefulPartitionedCall?
tf.__operators__.add_31/AddV2AddV2!tf.__operators__.add_30/AddV2:z:0-embedding_31/StatefulPartitionedCall:output:0*
T0*,
_output_shapes
:??????????2
tf.__operators__.add_31/AddV2?
 tf.expand_dims_10/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2"
 tf.expand_dims_10/ExpandDims/dim?
tf.expand_dims_10/ExpandDims
ExpandDimstf.cast_15/Cast:y:0)tf.expand_dims_10/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????2
tf.expand_dims_10/ExpandDims?
tf.math.multiply_10/MulMul!tf.__operators__.add_31/AddV2:z:0%tf.expand_dims_10/ExpandDims:output:0*
T0*,
_output_shapes
:??????????2
tf.math.multiply_10/Mul?
+tf.math.reduce_sum_10/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2-
+tf.math.reduce_sum_10/Sum/reduction_indices?
tf.math.reduce_sum_10/SumSumtf.math.multiply_10/Mul:z:04tf.math.reduce_sum_10/Sum/reduction_indices:output:0*
T0*(
_output_shapes
:??????????2
tf.math.reduce_sum_10/Sum?
IdentityIdentity"tf.math.reduce_sum_10/Sum:output:0%^embedding_30/StatefulPartitionedCall%^embedding_31/StatefulPartitionedCall%^embedding_32/StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:?????????: :::2L
$embedding_30/StatefulPartitionedCall$embedding_30/StatefulPartitionedCall2L
$embedding_31/StatefulPartitionedCall$embedding_31/StatefulPartitionedCall2L
$embedding_32/StatefulPartitionedCall$embedding_32/StatefulPartitionedCall:Q M
'
_output_shapes
:?????????
"
_user_specified_name
input_11:

_output_shapes
: 
?	
?
F__inference_dense_53_layer_call_and_return_conditional_losses_41027555

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
~
.__inference_normalize_5_layer_call_fn_41027545
x
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *R
fMRK
I__inference_normalize_5_layer_call_and_return_conditional_losses_410261292
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:K G
(
_output_shapes
:??????????

_user_specified_namex
?Y
?

L__inference_custom_model_5_layer_call_and_return_conditional_losses_41026172

cards0

cards1
bets+
'tf_math_greater_equal_17_greaterequal_y
model_10_41025841
model_10_41025843
model_10_41025845
model_10_41025847
model_11_41025876
model_11_41025878
model_11_41025880
model_11_41025882/
+tf_clip_by_value_17_clip_by_value_minimum_y'
#tf_clip_by_value_17_clip_by_value_y
dense_45_41025916
dense_45_41025918
dense_48_41025942
dense_48_41025944
dense_46_41025969
dense_46_41025971
dense_47_41025996
dense_47_41025998
dense_49_41026022
dense_49_41026024
dense_50_41026050
dense_50_41026052
dense_51_41026077
dense_51_41026079
dense_52_41026105
dense_52_41026107
normalize_5_41026140
normalize_5_41026142
dense_53_41026166
dense_53_41026168
identity?? dense_45/StatefulPartitionedCall? dense_46/StatefulPartitionedCall? dense_47/StatefulPartitionedCall? dense_48/StatefulPartitionedCall? dense_49/StatefulPartitionedCall? dense_50/StatefulPartitionedCall? dense_51/StatefulPartitionedCall? dense_52/StatefulPartitionedCall? dense_53/StatefulPartitionedCall? model_10/StatefulPartitionedCall? model_11/StatefulPartitionedCall?#normalize_5/StatefulPartitionedCall?
%tf.math.greater_equal_17/GreaterEqualGreaterEqualbets'tf_math_greater_equal_17_greaterequal_y*
T0*'
_output_shapes
:?????????
2'
%tf.math.greater_equal_17/GreaterEqual?
 model_10/StatefulPartitionedCallStatefulPartitionedCallcards0model_10_41025841model_10_41025843model_10_41025845model_10_41025847*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*%
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_model_10_layer_call_and_return_conditional_losses_410255252"
 model_10/StatefulPartitionedCall?
 model_11/StatefulPartitionedCallStatefulPartitionedCallcards1model_11_41025876model_11_41025878model_11_41025880model_11_41025882*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*%
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_model_11_layer_call_and_return_conditional_losses_410257512"
 model_11/StatefulPartitionedCall?
)tf.clip_by_value_17/clip_by_value/MinimumMinimumbets+tf_clip_by_value_17_clip_by_value_minimum_y*
T0*'
_output_shapes
:?????????
2+
)tf.clip_by_value_17/clip_by_value/Minimum?
!tf.clip_by_value_17/clip_by_valueMaximum-tf.clip_by_value_17/clip_by_value/Minimum:z:0#tf_clip_by_value_17_clip_by_value_y*
T0*'
_output_shapes
:?????????
2#
!tf.clip_by_value_17/clip_by_value?
tf.cast_17/CastCast)tf.math.greater_equal_17/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????
2
tf.cast_17/Castv
tf.concat_15/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
tf.concat_15/concat/axis?
tf.concat_15/concatConcatV2)model_10/StatefulPartitionedCall:output:0)model_11/StatefulPartitionedCall:output:0!tf.concat_15/concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????2
tf.concat_15/concat
tf.concat_16/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
tf.concat_16/concat/axis?
tf.concat_16/concatConcatV2%tf.clip_by_value_17/clip_by_value:z:0tf.cast_17/Cast:y:0!tf.concat_16/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????2
tf.concat_16/concat?
 dense_45/StatefulPartitionedCallStatefulPartitionedCalltf.concat_15/concat:output:0dense_45_41025916dense_45_41025918*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_dense_45_layer_call_and_return_conditional_losses_410259052"
 dense_45/StatefulPartitionedCall?
 dense_48/StatefulPartitionedCallStatefulPartitionedCalltf.concat_16/concat:output:0dense_48_41025942dense_48_41025944*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_dense_48_layer_call_and_return_conditional_losses_410259312"
 dense_48/StatefulPartitionedCall?
 dense_46/StatefulPartitionedCallStatefulPartitionedCall)dense_45/StatefulPartitionedCall:output:0dense_46_41025969dense_46_41025971*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_dense_46_layer_call_and_return_conditional_losses_410259582"
 dense_46/StatefulPartitionedCall?
 dense_47/StatefulPartitionedCallStatefulPartitionedCall)dense_46/StatefulPartitionedCall:output:0dense_47_41025996dense_47_41025998*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_dense_47_layer_call_and_return_conditional_losses_410259852"
 dense_47/StatefulPartitionedCall?
 dense_49/StatefulPartitionedCallStatefulPartitionedCall)dense_48/StatefulPartitionedCall:output:0dense_49_41026022dense_49_41026024*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_dense_49_layer_call_and_return_conditional_losses_410260112"
 dense_49/StatefulPartitionedCall
tf.concat_17/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
tf.concat_17/concat/axis?
tf.concat_17/concatConcatV2)dense_47/StatefulPartitionedCall:output:0)dense_49/StatefulPartitionedCall:output:0!tf.concat_17/concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????2
tf.concat_17/concat?
 dense_50/StatefulPartitionedCallStatefulPartitionedCalltf.concat_17/concat:output:0dense_50_41026050dense_50_41026052*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_dense_50_layer_call_and_return_conditional_losses_410260392"
 dense_50/StatefulPartitionedCall?
tf.nn.relu_15/ReluRelu)dense_50/StatefulPartitionedCall:output:0*
T0*(
_output_shapes
:??????????2
tf.nn.relu_15/Relu?
 dense_51/StatefulPartitionedCallStatefulPartitionedCall tf.nn.relu_15/Relu:activations:0dense_51_41026077dense_51_41026079*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_dense_51_layer_call_and_return_conditional_losses_410260662"
 dense_51/StatefulPartitionedCall?
tf.__operators__.add_34/AddV2AddV2)dense_51/StatefulPartitionedCall:output:0 tf.nn.relu_15/Relu:activations:0*
T0*(
_output_shapes
:??????????2
tf.__operators__.add_34/AddV2?
tf.nn.relu_16/ReluRelu!tf.__operators__.add_34/AddV2:z:0*
T0*(
_output_shapes
:??????????2
tf.nn.relu_16/Relu?
 dense_52/StatefulPartitionedCallStatefulPartitionedCall tf.nn.relu_16/Relu:activations:0dense_52_41026105dense_52_41026107*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_dense_52_layer_call_and_return_conditional_losses_410260942"
 dense_52/StatefulPartitionedCall?
tf.__operators__.add_35/AddV2AddV2)dense_52/StatefulPartitionedCall:output:0 tf.nn.relu_16/Relu:activations:0*
T0*(
_output_shapes
:??????????2
tf.__operators__.add_35/AddV2?
tf.nn.relu_17/ReluRelu!tf.__operators__.add_35/AddV2:z:0*
T0*(
_output_shapes
:??????????2
tf.nn.relu_17/Relu?
#normalize_5/StatefulPartitionedCallStatefulPartitionedCall tf.nn.relu_17/Relu:activations:0normalize_5_41026140normalize_5_41026142*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *R
fMRK
I__inference_normalize_5_layer_call_and_return_conditional_losses_410261292%
#normalize_5/StatefulPartitionedCall?
 dense_53/StatefulPartitionedCallStatefulPartitionedCall,normalize_5/StatefulPartitionedCall:output:0dense_53_41026166dense_53_41026168*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_dense_53_layer_call_and_return_conditional_losses_410261552"
 dense_53/StatefulPartitionedCall?
IdentityIdentity)dense_53/StatefulPartitionedCall:output:0!^dense_45/StatefulPartitionedCall!^dense_46/StatefulPartitionedCall!^dense_47/StatefulPartitionedCall!^dense_48/StatefulPartitionedCall!^dense_49/StatefulPartitionedCall!^dense_50/StatefulPartitionedCall!^dense_51/StatefulPartitionedCall!^dense_52/StatefulPartitionedCall!^dense_53/StatefulPartitionedCall!^model_10/StatefulPartitionedCall!^model_11/StatefulPartitionedCall$^normalize_5/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????:?????????:?????????
: : :::: :::: : ::::::::::::::::::::2D
 dense_45/StatefulPartitionedCall dense_45/StatefulPartitionedCall2D
 dense_46/StatefulPartitionedCall dense_46/StatefulPartitionedCall2D
 dense_47/StatefulPartitionedCall dense_47/StatefulPartitionedCall2D
 dense_48/StatefulPartitionedCall dense_48/StatefulPartitionedCall2D
 dense_49/StatefulPartitionedCall dense_49/StatefulPartitionedCall2D
 dense_50/StatefulPartitionedCall dense_50/StatefulPartitionedCall2D
 dense_51/StatefulPartitionedCall dense_51/StatefulPartitionedCall2D
 dense_52/StatefulPartitionedCall dense_52/StatefulPartitionedCall2D
 dense_53/StatefulPartitionedCall dense_53/StatefulPartitionedCall2D
 model_10/StatefulPartitionedCall model_10/StatefulPartitionedCall2D
 model_11/StatefulPartitionedCall model_11/StatefulPartitionedCall2J
#normalize_5/StatefulPartitionedCall#normalize_5/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_namecards0:OK
'
_output_shapes
:?????????
 
_user_specified_namecards1:MI
'
_output_shapes
:?????????


_user_specified_namebets:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?	
?
F__inference_dense_52_layer_call_and_return_conditional_losses_41027510

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
+__inference_model_10_layer_call_fn_41025536
input_11
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_11unknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*%
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_model_10_layer_call_and_return_conditional_losses_410255252
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:?????????: :::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:?????????
"
_user_specified_name
input_11:

_output_shapes
: 
??
?
L__inference_custom_model_5_layer_call_and_return_conditional_losses_41026836

inputs_0_0

inputs_0_1
inputs_1+
'tf_math_greater_equal_17_greaterequal_y4
0model_10_tf_math_greater_equal_15_greaterequal_y3
/model_10_embedding_32_embedding_lookup_410266863
/model_10_embedding_30_embedding_lookup_410266923
/model_10_embedding_31_embedding_lookup_410267004
0model_11_tf_math_greater_equal_16_greaterequal_y3
/model_11_embedding_35_embedding_lookup_410267243
/model_11_embedding_33_embedding_lookup_410267303
/model_11_embedding_34_embedding_lookup_41026738/
+tf_clip_by_value_17_clip_by_value_minimum_y'
#tf_clip_by_value_17_clip_by_value_y+
'dense_45_matmul_readvariableop_resource,
(dense_45_biasadd_readvariableop_resource+
'dense_48_matmul_readvariableop_resource,
(dense_48_biasadd_readvariableop_resource+
'dense_46_matmul_readvariableop_resource,
(dense_46_biasadd_readvariableop_resource+
'dense_47_matmul_readvariableop_resource,
(dense_47_biasadd_readvariableop_resource+
'dense_49_matmul_readvariableop_resource,
(dense_49_biasadd_readvariableop_resource+
'dense_50_matmul_readvariableop_resource,
(dense_50_biasadd_readvariableop_resource+
'dense_51_matmul_readvariableop_resource,
(dense_51_biasadd_readvariableop_resource+
'dense_52_matmul_readvariableop_resource,
(dense_52_biasadd_readvariableop_resource?
;normalize_5_normalization_5_reshape_readvariableop_resourceA
=normalize_5_normalization_5_reshape_1_readvariableop_resource+
'dense_53_matmul_readvariableop_resource,
(dense_53_biasadd_readvariableop_resource
identity??dense_45/BiasAdd/ReadVariableOp?dense_45/MatMul/ReadVariableOp?dense_46/BiasAdd/ReadVariableOp?dense_46/MatMul/ReadVariableOp?dense_47/BiasAdd/ReadVariableOp?dense_47/MatMul/ReadVariableOp?dense_48/BiasAdd/ReadVariableOp?dense_48/MatMul/ReadVariableOp?dense_49/BiasAdd/ReadVariableOp?dense_49/MatMul/ReadVariableOp?dense_50/BiasAdd/ReadVariableOp?dense_50/MatMul/ReadVariableOp?dense_51/BiasAdd/ReadVariableOp?dense_51/MatMul/ReadVariableOp?dense_52/BiasAdd/ReadVariableOp?dense_52/MatMul/ReadVariableOp?dense_53/BiasAdd/ReadVariableOp?dense_53/MatMul/ReadVariableOp?&model_10/embedding_30/embedding_lookup?&model_10/embedding_31/embedding_lookup?&model_10/embedding_32/embedding_lookup?&model_11/embedding_33/embedding_lookup?&model_11/embedding_34/embedding_lookup?&model_11/embedding_35/embedding_lookup?2normalize_5/normalization_5/Reshape/ReadVariableOp?4normalize_5/normalization_5/Reshape_1/ReadVariableOp?
%tf.math.greater_equal_17/GreaterEqualGreaterEqualinputs_1'tf_math_greater_equal_17_greaterequal_y*
T0*'
_output_shapes
:?????????
2'
%tf.math.greater_equal_17/GreaterEqual?
model_10/flatten_10/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
model_10/flatten_10/Const?
model_10/flatten_10/ReshapeReshape
inputs_0_0"model_10/flatten_10/Const:output:0*
T0*'
_output_shapes
:?????????2
model_10/flatten_10/Reshape?
4model_10/tf.clip_by_value_15/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 * $tI26
4model_10/tf.clip_by_value_15/clip_by_value/Minimum/y?
2model_10/tf.clip_by_value_15/clip_by_value/MinimumMinimum$model_10/flatten_10/Reshape:output:0=model_10/tf.clip_by_value_15/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????24
2model_10/tf.clip_by_value_15/clip_by_value/Minimum?
,model_10/tf.clip_by_value_15/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2.
,model_10/tf.clip_by_value_15/clip_by_value/y?
*model_10/tf.clip_by_value_15/clip_by_valueMaximum6model_10/tf.clip_by_value_15/clip_by_value/Minimum:z:05model_10/tf.clip_by_value_15/clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????2,
*model_10/tf.clip_by_value_15/clip_by_value?
-model_10/tf.compat.v1.floor_div_10/FloorDiv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2/
-model_10/tf.compat.v1.floor_div_10/FloorDiv/y?
+model_10/tf.compat.v1.floor_div_10/FloorDivFloorDiv.model_10/tf.clip_by_value_15/clip_by_value:z:06model_10/tf.compat.v1.floor_div_10/FloorDiv/y:output:0*
T0*'
_output_shapes
:?????????2-
+model_10/tf.compat.v1.floor_div_10/FloorDiv?
.model_10/tf.math.greater_equal_15/GreaterEqualGreaterEqual$model_10/flatten_10/Reshape:output:00model_10_tf_math_greater_equal_15_greaterequal_y*
T0*'
_output_shapes
:?????????20
.model_10/tf.math.greater_equal_15/GreaterEqual?
'model_10/tf.math.floormod_10/FloorMod/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2)
'model_10/tf.math.floormod_10/FloorMod/y?
%model_10/tf.math.floormod_10/FloorModFloorMod.model_10/tf.clip_by_value_15/clip_by_value:z:00model_10/tf.math.floormod_10/FloorMod/y:output:0*
T0*'
_output_shapes
:?????????2'
%model_10/tf.math.floormod_10/FloorMod?
model_10/embedding_32/CastCast.model_10/tf.clip_by_value_15/clip_by_value:z:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2
model_10/embedding_32/Cast?
&model_10/embedding_32/embedding_lookupResourceGather/model_10_embedding_32_embedding_lookup_41026686model_10/embedding_32/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*B
_class8
64loc:@model_10/embedding_32/embedding_lookup/41026686*,
_output_shapes
:??????????*
dtype02(
&model_10/embedding_32/embedding_lookup?
/model_10/embedding_32/embedding_lookup/IdentityIdentity/model_10/embedding_32/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*B
_class8
64loc:@model_10/embedding_32/embedding_lookup/41026686*,
_output_shapes
:??????????21
/model_10/embedding_32/embedding_lookup/Identity?
1model_10/embedding_32/embedding_lookup/Identity_1Identity8model_10/embedding_32/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????23
1model_10/embedding_32/embedding_lookup/Identity_1?
model_10/embedding_30/CastCast/model_10/tf.compat.v1.floor_div_10/FloorDiv:z:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2
model_10/embedding_30/Cast?
&model_10/embedding_30/embedding_lookupResourceGather/model_10_embedding_30_embedding_lookup_41026692model_10/embedding_30/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*B
_class8
64loc:@model_10/embedding_30/embedding_lookup/41026692*,
_output_shapes
:??????????*
dtype02(
&model_10/embedding_30/embedding_lookup?
/model_10/embedding_30/embedding_lookup/IdentityIdentity/model_10/embedding_30/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*B
_class8
64loc:@model_10/embedding_30/embedding_lookup/41026692*,
_output_shapes
:??????????21
/model_10/embedding_30/embedding_lookup/Identity?
1model_10/embedding_30/embedding_lookup/Identity_1Identity8model_10/embedding_30/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????23
1model_10/embedding_30/embedding_lookup/Identity_1?
model_10/tf.cast_15/CastCast2model_10/tf.math.greater_equal_15/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2
model_10/tf.cast_15/Cast?
&model_10/tf.__operators__.add_30/AddV2AddV2:model_10/embedding_32/embedding_lookup/Identity_1:output:0:model_10/embedding_30/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:??????????2(
&model_10/tf.__operators__.add_30/AddV2?
model_10/embedding_31/CastCast)model_10/tf.math.floormod_10/FloorMod:z:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2
model_10/embedding_31/Cast?
&model_10/embedding_31/embedding_lookupResourceGather/model_10_embedding_31_embedding_lookup_41026700model_10/embedding_31/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*B
_class8
64loc:@model_10/embedding_31/embedding_lookup/41026700*,
_output_shapes
:??????????*
dtype02(
&model_10/embedding_31/embedding_lookup?
/model_10/embedding_31/embedding_lookup/IdentityIdentity/model_10/embedding_31/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*B
_class8
64loc:@model_10/embedding_31/embedding_lookup/41026700*,
_output_shapes
:??????????21
/model_10/embedding_31/embedding_lookup/Identity?
1model_10/embedding_31/embedding_lookup/Identity_1Identity8model_10/embedding_31/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????23
1model_10/embedding_31/embedding_lookup/Identity_1?
&model_10/tf.__operators__.add_31/AddV2AddV2*model_10/tf.__operators__.add_30/AddV2:z:0:model_10/embedding_31/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:??????????2(
&model_10/tf.__operators__.add_31/AddV2?
)model_10/tf.expand_dims_10/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2+
)model_10/tf.expand_dims_10/ExpandDims/dim?
%model_10/tf.expand_dims_10/ExpandDims
ExpandDimsmodel_10/tf.cast_15/Cast:y:02model_10/tf.expand_dims_10/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????2'
%model_10/tf.expand_dims_10/ExpandDims?
 model_10/tf.math.multiply_10/MulMul*model_10/tf.__operators__.add_31/AddV2:z:0.model_10/tf.expand_dims_10/ExpandDims:output:0*
T0*,
_output_shapes
:??????????2"
 model_10/tf.math.multiply_10/Mul?
4model_10/tf.math.reduce_sum_10/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :26
4model_10/tf.math.reduce_sum_10/Sum/reduction_indices?
"model_10/tf.math.reduce_sum_10/SumSum$model_10/tf.math.multiply_10/Mul:z:0=model_10/tf.math.reduce_sum_10/Sum/reduction_indices:output:0*
T0*(
_output_shapes
:??????????2$
"model_10/tf.math.reduce_sum_10/Sum?
model_11/flatten_11/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
model_11/flatten_11/Const?
model_11/flatten_11/ReshapeReshape
inputs_0_1"model_11/flatten_11/Const:output:0*
T0*'
_output_shapes
:?????????2
model_11/flatten_11/Reshape?
4model_11/tf.clip_by_value_16/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 * $tI26
4model_11/tf.clip_by_value_16/clip_by_value/Minimum/y?
2model_11/tf.clip_by_value_16/clip_by_value/MinimumMinimum$model_11/flatten_11/Reshape:output:0=model_11/tf.clip_by_value_16/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????24
2model_11/tf.clip_by_value_16/clip_by_value/Minimum?
,model_11/tf.clip_by_value_16/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2.
,model_11/tf.clip_by_value_16/clip_by_value/y?
*model_11/tf.clip_by_value_16/clip_by_valueMaximum6model_11/tf.clip_by_value_16/clip_by_value/Minimum:z:05model_11/tf.clip_by_value_16/clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????2,
*model_11/tf.clip_by_value_16/clip_by_value?
-model_11/tf.compat.v1.floor_div_11/FloorDiv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2/
-model_11/tf.compat.v1.floor_div_11/FloorDiv/y?
+model_11/tf.compat.v1.floor_div_11/FloorDivFloorDiv.model_11/tf.clip_by_value_16/clip_by_value:z:06model_11/tf.compat.v1.floor_div_11/FloorDiv/y:output:0*
T0*'
_output_shapes
:?????????2-
+model_11/tf.compat.v1.floor_div_11/FloorDiv?
.model_11/tf.math.greater_equal_16/GreaterEqualGreaterEqual$model_11/flatten_11/Reshape:output:00model_11_tf_math_greater_equal_16_greaterequal_y*
T0*'
_output_shapes
:?????????20
.model_11/tf.math.greater_equal_16/GreaterEqual?
'model_11/tf.math.floormod_11/FloorMod/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2)
'model_11/tf.math.floormod_11/FloorMod/y?
%model_11/tf.math.floormod_11/FloorModFloorMod.model_11/tf.clip_by_value_16/clip_by_value:z:00model_11/tf.math.floormod_11/FloorMod/y:output:0*
T0*'
_output_shapes
:?????????2'
%model_11/tf.math.floormod_11/FloorMod?
model_11/embedding_35/CastCast.model_11/tf.clip_by_value_16/clip_by_value:z:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2
model_11/embedding_35/Cast?
&model_11/embedding_35/embedding_lookupResourceGather/model_11_embedding_35_embedding_lookup_41026724model_11/embedding_35/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*B
_class8
64loc:@model_11/embedding_35/embedding_lookup/41026724*,
_output_shapes
:??????????*
dtype02(
&model_11/embedding_35/embedding_lookup?
/model_11/embedding_35/embedding_lookup/IdentityIdentity/model_11/embedding_35/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*B
_class8
64loc:@model_11/embedding_35/embedding_lookup/41026724*,
_output_shapes
:??????????21
/model_11/embedding_35/embedding_lookup/Identity?
1model_11/embedding_35/embedding_lookup/Identity_1Identity8model_11/embedding_35/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????23
1model_11/embedding_35/embedding_lookup/Identity_1?
model_11/embedding_33/CastCast/model_11/tf.compat.v1.floor_div_11/FloorDiv:z:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2
model_11/embedding_33/Cast?
&model_11/embedding_33/embedding_lookupResourceGather/model_11_embedding_33_embedding_lookup_41026730model_11/embedding_33/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*B
_class8
64loc:@model_11/embedding_33/embedding_lookup/41026730*,
_output_shapes
:??????????*
dtype02(
&model_11/embedding_33/embedding_lookup?
/model_11/embedding_33/embedding_lookup/IdentityIdentity/model_11/embedding_33/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*B
_class8
64loc:@model_11/embedding_33/embedding_lookup/41026730*,
_output_shapes
:??????????21
/model_11/embedding_33/embedding_lookup/Identity?
1model_11/embedding_33/embedding_lookup/Identity_1Identity8model_11/embedding_33/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????23
1model_11/embedding_33/embedding_lookup/Identity_1?
model_11/tf.cast_16/CastCast2model_11/tf.math.greater_equal_16/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2
model_11/tf.cast_16/Cast?
&model_11/tf.__operators__.add_32/AddV2AddV2:model_11/embedding_35/embedding_lookup/Identity_1:output:0:model_11/embedding_33/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:??????????2(
&model_11/tf.__operators__.add_32/AddV2?
model_11/embedding_34/CastCast)model_11/tf.math.floormod_11/FloorMod:z:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2
model_11/embedding_34/Cast?
&model_11/embedding_34/embedding_lookupResourceGather/model_11_embedding_34_embedding_lookup_41026738model_11/embedding_34/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*B
_class8
64loc:@model_11/embedding_34/embedding_lookup/41026738*,
_output_shapes
:??????????*
dtype02(
&model_11/embedding_34/embedding_lookup?
/model_11/embedding_34/embedding_lookup/IdentityIdentity/model_11/embedding_34/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*B
_class8
64loc:@model_11/embedding_34/embedding_lookup/41026738*,
_output_shapes
:??????????21
/model_11/embedding_34/embedding_lookup/Identity?
1model_11/embedding_34/embedding_lookup/Identity_1Identity8model_11/embedding_34/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????23
1model_11/embedding_34/embedding_lookup/Identity_1?
&model_11/tf.__operators__.add_33/AddV2AddV2*model_11/tf.__operators__.add_32/AddV2:z:0:model_11/embedding_34/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:??????????2(
&model_11/tf.__operators__.add_33/AddV2?
)model_11/tf.expand_dims_11/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2+
)model_11/tf.expand_dims_11/ExpandDims/dim?
%model_11/tf.expand_dims_11/ExpandDims
ExpandDimsmodel_11/tf.cast_16/Cast:y:02model_11/tf.expand_dims_11/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????2'
%model_11/tf.expand_dims_11/ExpandDims?
 model_11/tf.math.multiply_11/MulMul*model_11/tf.__operators__.add_33/AddV2:z:0.model_11/tf.expand_dims_11/ExpandDims:output:0*
T0*,
_output_shapes
:??????????2"
 model_11/tf.math.multiply_11/Mul?
4model_11/tf.math.reduce_sum_11/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :26
4model_11/tf.math.reduce_sum_11/Sum/reduction_indices?
"model_11/tf.math.reduce_sum_11/SumSum$model_11/tf.math.multiply_11/Mul:z:0=model_11/tf.math.reduce_sum_11/Sum/reduction_indices:output:0*
T0*(
_output_shapes
:??????????2$
"model_11/tf.math.reduce_sum_11/Sum?
)tf.clip_by_value_17/clip_by_value/MinimumMinimuminputs_1+tf_clip_by_value_17_clip_by_value_minimum_y*
T0*'
_output_shapes
:?????????
2+
)tf.clip_by_value_17/clip_by_value/Minimum?
!tf.clip_by_value_17/clip_by_valueMaximum-tf.clip_by_value_17/clip_by_value/Minimum:z:0#tf_clip_by_value_17_clip_by_value_y*
T0*'
_output_shapes
:?????????
2#
!tf.clip_by_value_17/clip_by_value?
tf.cast_17/CastCast)tf.math.greater_equal_17/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????
2
tf.cast_17/Castv
tf.concat_15/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
tf.concat_15/concat/axis?
tf.concat_15/concatConcatV2+model_10/tf.math.reduce_sum_10/Sum:output:0+model_11/tf.math.reduce_sum_11/Sum:output:0!tf.concat_15/concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????2
tf.concat_15/concat
tf.concat_16/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
tf.concat_16/concat/axis?
tf.concat_16/concatConcatV2%tf.clip_by_value_17/clip_by_value:z:0tf.cast_17/Cast:y:0!tf.concat_16/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????2
tf.concat_16/concat?
dense_45/MatMul/ReadVariableOpReadVariableOp'dense_45_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02 
dense_45/MatMul/ReadVariableOp?
dense_45/MatMulMatMultf.concat_15/concat:output:0&dense_45/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_45/MatMul?
dense_45/BiasAdd/ReadVariableOpReadVariableOp(dense_45_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
dense_45/BiasAdd/ReadVariableOp?
dense_45/BiasAddBiasAdddense_45/MatMul:product:0'dense_45/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_45/BiasAddt
dense_45/ReluReludense_45/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_45/Relu?
dense_48/MatMul/ReadVariableOpReadVariableOp'dense_48_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02 
dense_48/MatMul/ReadVariableOp?
dense_48/MatMulMatMultf.concat_16/concat:output:0&dense_48/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_48/MatMul?
dense_48/BiasAdd/ReadVariableOpReadVariableOp(dense_48_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
dense_48/BiasAdd/ReadVariableOp?
dense_48/BiasAddBiasAdddense_48/MatMul:product:0'dense_48/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_48/BiasAdd?
dense_46/MatMul/ReadVariableOpReadVariableOp'dense_46_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02 
dense_46/MatMul/ReadVariableOp?
dense_46/MatMulMatMuldense_45/Relu:activations:0&dense_46/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_46/MatMul?
dense_46/BiasAdd/ReadVariableOpReadVariableOp(dense_46_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
dense_46/BiasAdd/ReadVariableOp?
dense_46/BiasAddBiasAdddense_46/MatMul:product:0'dense_46/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_46/BiasAddt
dense_46/ReluReludense_46/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_46/Relu?
dense_47/MatMul/ReadVariableOpReadVariableOp'dense_47_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02 
dense_47/MatMul/ReadVariableOp?
dense_47/MatMulMatMuldense_46/Relu:activations:0&dense_47/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_47/MatMul?
dense_47/BiasAdd/ReadVariableOpReadVariableOp(dense_47_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
dense_47/BiasAdd/ReadVariableOp?
dense_47/BiasAddBiasAdddense_47/MatMul:product:0'dense_47/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_47/BiasAddt
dense_47/ReluReludense_47/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_47/Relu?
dense_49/MatMul/ReadVariableOpReadVariableOp'dense_49_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02 
dense_49/MatMul/ReadVariableOp?
dense_49/MatMulMatMuldense_48/BiasAdd:output:0&dense_49/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_49/MatMul?
dense_49/BiasAdd/ReadVariableOpReadVariableOp(dense_49_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
dense_49/BiasAdd/ReadVariableOp?
dense_49/BiasAddBiasAdddense_49/MatMul:product:0'dense_49/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_49/BiasAdd
tf.concat_17/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
tf.concat_17/concat/axis?
tf.concat_17/concatConcatV2dense_47/Relu:activations:0dense_49/BiasAdd:output:0!tf.concat_17/concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????2
tf.concat_17/concat?
dense_50/MatMul/ReadVariableOpReadVariableOp'dense_50_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02 
dense_50/MatMul/ReadVariableOp?
dense_50/MatMulMatMultf.concat_17/concat:output:0&dense_50/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_50/MatMul?
dense_50/BiasAdd/ReadVariableOpReadVariableOp(dense_50_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
dense_50/BiasAdd/ReadVariableOp?
dense_50/BiasAddBiasAdddense_50/MatMul:product:0'dense_50/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_50/BiasAdd~
tf.nn.relu_15/ReluReludense_50/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
tf.nn.relu_15/Relu?
dense_51/MatMul/ReadVariableOpReadVariableOp'dense_51_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02 
dense_51/MatMul/ReadVariableOp?
dense_51/MatMulMatMul tf.nn.relu_15/Relu:activations:0&dense_51/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_51/MatMul?
dense_51/BiasAdd/ReadVariableOpReadVariableOp(dense_51_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
dense_51/BiasAdd/ReadVariableOp?
dense_51/BiasAddBiasAdddense_51/MatMul:product:0'dense_51/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_51/BiasAdd?
tf.__operators__.add_34/AddV2AddV2dense_51/BiasAdd:output:0 tf.nn.relu_15/Relu:activations:0*
T0*(
_output_shapes
:??????????2
tf.__operators__.add_34/AddV2?
tf.nn.relu_16/ReluRelu!tf.__operators__.add_34/AddV2:z:0*
T0*(
_output_shapes
:??????????2
tf.nn.relu_16/Relu?
dense_52/MatMul/ReadVariableOpReadVariableOp'dense_52_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02 
dense_52/MatMul/ReadVariableOp?
dense_52/MatMulMatMul tf.nn.relu_16/Relu:activations:0&dense_52/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_52/MatMul?
dense_52/BiasAdd/ReadVariableOpReadVariableOp(dense_52_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
dense_52/BiasAdd/ReadVariableOp?
dense_52/BiasAddBiasAdddense_52/MatMul:product:0'dense_52/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_52/BiasAdd?
tf.__operators__.add_35/AddV2AddV2dense_52/BiasAdd:output:0 tf.nn.relu_16/Relu:activations:0*
T0*(
_output_shapes
:??????????2
tf.__operators__.add_35/AddV2?
tf.nn.relu_17/ReluRelu!tf.__operators__.add_35/AddV2:z:0*
T0*(
_output_shapes
:??????????2
tf.nn.relu_17/Relu?
2normalize_5/normalization_5/Reshape/ReadVariableOpReadVariableOp;normalize_5_normalization_5_reshape_readvariableop_resource*
_output_shapes	
:?*
dtype024
2normalize_5/normalization_5/Reshape/ReadVariableOp?
)normalize_5/normalization_5/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2+
)normalize_5/normalization_5/Reshape/shape?
#normalize_5/normalization_5/ReshapeReshape:normalize_5/normalization_5/Reshape/ReadVariableOp:value:02normalize_5/normalization_5/Reshape/shape:output:0*
T0*
_output_shapes
:	?2%
#normalize_5/normalization_5/Reshape?
4normalize_5/normalization_5/Reshape_1/ReadVariableOpReadVariableOp=normalize_5_normalization_5_reshape_1_readvariableop_resource*
_output_shapes	
:?*
dtype026
4normalize_5/normalization_5/Reshape_1/ReadVariableOp?
+normalize_5/normalization_5/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2-
+normalize_5/normalization_5/Reshape_1/shape?
%normalize_5/normalization_5/Reshape_1Reshape<normalize_5/normalization_5/Reshape_1/ReadVariableOp:value:04normalize_5/normalization_5/Reshape_1/shape:output:0*
T0*
_output_shapes
:	?2'
%normalize_5/normalization_5/Reshape_1?
normalize_5/normalization_5/subSub tf.nn.relu_17/Relu:activations:0,normalize_5/normalization_5/Reshape:output:0*
T0*(
_output_shapes
:??????????2!
normalize_5/normalization_5/sub?
 normalize_5/normalization_5/SqrtSqrt.normalize_5/normalization_5/Reshape_1:output:0*
T0*
_output_shapes
:	?2"
 normalize_5/normalization_5/Sqrt?
%normalize_5/normalization_5/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32'
%normalize_5/normalization_5/Maximum/y?
#normalize_5/normalization_5/MaximumMaximum$normalize_5/normalization_5/Sqrt:y:0.normalize_5/normalization_5/Maximum/y:output:0*
T0*
_output_shapes
:	?2%
#normalize_5/normalization_5/Maximum?
#normalize_5/normalization_5/truedivRealDiv#normalize_5/normalization_5/sub:z:0'normalize_5/normalization_5/Maximum:z:0*
T0*(
_output_shapes
:??????????2%
#normalize_5/normalization_5/truediv?
dense_53/MatMul/ReadVariableOpReadVariableOp'dense_53_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02 
dense_53/MatMul/ReadVariableOp?
dense_53/MatMulMatMul'normalize_5/normalization_5/truediv:z:0&dense_53/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_53/MatMul?
dense_53/BiasAdd/ReadVariableOpReadVariableOp(dense_53_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_53/BiasAdd/ReadVariableOp?
dense_53/BiasAddBiasAdddense_53/MatMul:product:0'dense_53/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_53/BiasAdd?
IdentityIdentitydense_53/BiasAdd:output:0 ^dense_45/BiasAdd/ReadVariableOp^dense_45/MatMul/ReadVariableOp ^dense_46/BiasAdd/ReadVariableOp^dense_46/MatMul/ReadVariableOp ^dense_47/BiasAdd/ReadVariableOp^dense_47/MatMul/ReadVariableOp ^dense_48/BiasAdd/ReadVariableOp^dense_48/MatMul/ReadVariableOp ^dense_49/BiasAdd/ReadVariableOp^dense_49/MatMul/ReadVariableOp ^dense_50/BiasAdd/ReadVariableOp^dense_50/MatMul/ReadVariableOp ^dense_51/BiasAdd/ReadVariableOp^dense_51/MatMul/ReadVariableOp ^dense_52/BiasAdd/ReadVariableOp^dense_52/MatMul/ReadVariableOp ^dense_53/BiasAdd/ReadVariableOp^dense_53/MatMul/ReadVariableOp'^model_10/embedding_30/embedding_lookup'^model_10/embedding_31/embedding_lookup'^model_10/embedding_32/embedding_lookup'^model_11/embedding_33/embedding_lookup'^model_11/embedding_34/embedding_lookup'^model_11/embedding_35/embedding_lookup3^normalize_5/normalization_5/Reshape/ReadVariableOp5^normalize_5/normalization_5/Reshape_1/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????:?????????:?????????
: : :::: :::: : ::::::::::::::::::::2B
dense_45/BiasAdd/ReadVariableOpdense_45/BiasAdd/ReadVariableOp2@
dense_45/MatMul/ReadVariableOpdense_45/MatMul/ReadVariableOp2B
dense_46/BiasAdd/ReadVariableOpdense_46/BiasAdd/ReadVariableOp2@
dense_46/MatMul/ReadVariableOpdense_46/MatMul/ReadVariableOp2B
dense_47/BiasAdd/ReadVariableOpdense_47/BiasAdd/ReadVariableOp2@
dense_47/MatMul/ReadVariableOpdense_47/MatMul/ReadVariableOp2B
dense_48/BiasAdd/ReadVariableOpdense_48/BiasAdd/ReadVariableOp2@
dense_48/MatMul/ReadVariableOpdense_48/MatMul/ReadVariableOp2B
dense_49/BiasAdd/ReadVariableOpdense_49/BiasAdd/ReadVariableOp2@
dense_49/MatMul/ReadVariableOpdense_49/MatMul/ReadVariableOp2B
dense_50/BiasAdd/ReadVariableOpdense_50/BiasAdd/ReadVariableOp2@
dense_50/MatMul/ReadVariableOpdense_50/MatMul/ReadVariableOp2B
dense_51/BiasAdd/ReadVariableOpdense_51/BiasAdd/ReadVariableOp2@
dense_51/MatMul/ReadVariableOpdense_51/MatMul/ReadVariableOp2B
dense_52/BiasAdd/ReadVariableOpdense_52/BiasAdd/ReadVariableOp2@
dense_52/MatMul/ReadVariableOpdense_52/MatMul/ReadVariableOp2B
dense_53/BiasAdd/ReadVariableOpdense_53/BiasAdd/ReadVariableOp2@
dense_53/MatMul/ReadVariableOpdense_53/MatMul/ReadVariableOp2P
&model_10/embedding_30/embedding_lookup&model_10/embedding_30/embedding_lookup2P
&model_10/embedding_31/embedding_lookup&model_10/embedding_31/embedding_lookup2P
&model_10/embedding_32/embedding_lookup&model_10/embedding_32/embedding_lookup2P
&model_11/embedding_33/embedding_lookup&model_11/embedding_33/embedding_lookup2P
&model_11/embedding_34/embedding_lookup&model_11/embedding_34/embedding_lookup2P
&model_11/embedding_35/embedding_lookup&model_11/embedding_35/embedding_lookup2h
2normalize_5/normalization_5/Reshape/ReadVariableOp2normalize_5/normalization_5/Reshape/ReadVariableOp2l
4normalize_5/normalization_5/Reshape_1/ReadVariableOp4normalize_5/normalization_5/Reshape_1/ReadVariableOp:S O
'
_output_shapes
:?????????
$
_user_specified_name
inputs/0/0:SO
'
_output_shapes
:?????????
$
_user_specified_name
inputs/0/1:QM
'
_output_shapes
:?????????

"
_user_specified_name
inputs/1:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
??
?,
$__inference__traced_restore_41028220
file_prefix$
 assignvariableop_dense_45_kernel$
 assignvariableop_1_dense_45_bias&
"assignvariableop_2_dense_46_kernel$
 assignvariableop_3_dense_46_bias&
"assignvariableop_4_dense_48_kernel$
 assignvariableop_5_dense_48_bias&
"assignvariableop_6_dense_47_kernel$
 assignvariableop_7_dense_47_bias&
"assignvariableop_8_dense_49_kernel$
 assignvariableop_9_dense_49_bias'
#assignvariableop_10_dense_50_kernel%
!assignvariableop_11_dense_50_bias'
#assignvariableop_12_dense_51_kernel%
!assignvariableop_13_dense_51_bias'
#assignvariableop_14_dense_52_kernel%
!assignvariableop_15_dense_52_bias'
#assignvariableop_16_dense_53_kernel%
!assignvariableop_17_dense_53_bias!
assignvariableop_18_adam_iter#
assignvariableop_19_adam_beta_1#
assignvariableop_20_adam_beta_2"
assignvariableop_21_adam_decay*
&assignvariableop_22_adam_learning_rate/
+assignvariableop_23_embedding_32_embeddings/
+assignvariableop_24_embedding_30_embeddings/
+assignvariableop_25_embedding_31_embeddings/
+assignvariableop_26_embedding_35_embeddings/
+assignvariableop_27_embedding_33_embeddings/
+assignvariableop_28_embedding_34_embeddings8
4assignvariableop_29_normalize_5_normalization_5_mean<
8assignvariableop_30_normalize_5_normalization_5_variance9
5assignvariableop_31_normalize_5_normalization_5_count
assignvariableop_32_total
assignvariableop_33_count.
*assignvariableop_34_adam_dense_45_kernel_m,
(assignvariableop_35_adam_dense_45_bias_m.
*assignvariableop_36_adam_dense_46_kernel_m,
(assignvariableop_37_adam_dense_46_bias_m.
*assignvariableop_38_adam_dense_48_kernel_m,
(assignvariableop_39_adam_dense_48_bias_m.
*assignvariableop_40_adam_dense_47_kernel_m,
(assignvariableop_41_adam_dense_47_bias_m.
*assignvariableop_42_adam_dense_49_kernel_m,
(assignvariableop_43_adam_dense_49_bias_m.
*assignvariableop_44_adam_dense_50_kernel_m,
(assignvariableop_45_adam_dense_50_bias_m.
*assignvariableop_46_adam_dense_51_kernel_m,
(assignvariableop_47_adam_dense_51_bias_m.
*assignvariableop_48_adam_dense_52_kernel_m,
(assignvariableop_49_adam_dense_52_bias_m.
*assignvariableop_50_adam_dense_53_kernel_m,
(assignvariableop_51_adam_dense_53_bias_m6
2assignvariableop_52_adam_embedding_32_embeddings_m6
2assignvariableop_53_adam_embedding_30_embeddings_m6
2assignvariableop_54_adam_embedding_31_embeddings_m6
2assignvariableop_55_adam_embedding_35_embeddings_m6
2assignvariableop_56_adam_embedding_33_embeddings_m6
2assignvariableop_57_adam_embedding_34_embeddings_m.
*assignvariableop_58_adam_dense_45_kernel_v,
(assignvariableop_59_adam_dense_45_bias_v.
*assignvariableop_60_adam_dense_46_kernel_v,
(assignvariableop_61_adam_dense_46_bias_v.
*assignvariableop_62_adam_dense_48_kernel_v,
(assignvariableop_63_adam_dense_48_bias_v.
*assignvariableop_64_adam_dense_47_kernel_v,
(assignvariableop_65_adam_dense_47_bias_v.
*assignvariableop_66_adam_dense_49_kernel_v,
(assignvariableop_67_adam_dense_49_bias_v.
*assignvariableop_68_adam_dense_50_kernel_v,
(assignvariableop_69_adam_dense_50_bias_v.
*assignvariableop_70_adam_dense_51_kernel_v,
(assignvariableop_71_adam_dense_51_bias_v.
*assignvariableop_72_adam_dense_52_kernel_v,
(assignvariableop_73_adam_dense_52_bias_v.
*assignvariableop_74_adam_dense_53_kernel_v,
(assignvariableop_75_adam_dense_53_bias_v6
2assignvariableop_76_adam_embedding_32_embeddings_v6
2assignvariableop_77_adam_embedding_30_embeddings_v6
2assignvariableop_78_adam_embedding_31_embeddings_v6
2assignvariableop_79_adam_embedding_35_embeddings_v6
2assignvariableop_80_adam_embedding_33_embeddings_v6
2assignvariableop_81_adam_embedding_34_embeddings_v
identity_83??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_39?AssignVariableOp_4?AssignVariableOp_40?AssignVariableOp_41?AssignVariableOp_42?AssignVariableOp_43?AssignVariableOp_44?AssignVariableOp_45?AssignVariableOp_46?AssignVariableOp_47?AssignVariableOp_48?AssignVariableOp_49?AssignVariableOp_5?AssignVariableOp_50?AssignVariableOp_51?AssignVariableOp_52?AssignVariableOp_53?AssignVariableOp_54?AssignVariableOp_55?AssignVariableOp_56?AssignVariableOp_57?AssignVariableOp_58?AssignVariableOp_59?AssignVariableOp_6?AssignVariableOp_60?AssignVariableOp_61?AssignVariableOp_62?AssignVariableOp_63?AssignVariableOp_64?AssignVariableOp_65?AssignVariableOp_66?AssignVariableOp_67?AssignVariableOp_68?AssignVariableOp_69?AssignVariableOp_7?AssignVariableOp_70?AssignVariableOp_71?AssignVariableOp_72?AssignVariableOp_73?AssignVariableOp_74?AssignVariableOp_75?AssignVariableOp_76?AssignVariableOp_77?AssignVariableOp_78?AssignVariableOp_79?AssignVariableOp_8?AssignVariableOp_80?AssignVariableOp_81?AssignVariableOp_9?-
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:S*
dtype0*?,
value?,B?,SB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:S*
dtype0*?
value?B?SB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*a
dtypesW
U2S		2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOp assignvariableop_dense_45_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp assignvariableop_1_dense_45_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp"assignvariableop_2_dense_46_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp assignvariableop_3_dense_46_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp"assignvariableop_4_dense_48_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp assignvariableop_5_dense_48_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp"assignvariableop_6_dense_47_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp assignvariableop_7_dense_47_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp"assignvariableop_8_dense_49_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp assignvariableop_9_dense_49_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp#assignvariableop_10_dense_50_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOp!assignvariableop_11_dense_50_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOp#assignvariableop_12_dense_51_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOp!assignvariableop_13_dense_51_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOp#assignvariableop_14_dense_52_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOp!assignvariableop_15_dense_52_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOp#assignvariableop_16_dense_53_kernelIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOp!assignvariableop_17_dense_53_biasIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOpassignvariableop_18_adam_iterIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOpassignvariableop_19_adam_beta_1Identity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOpassignvariableop_20_adam_beta_2Identity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOpassignvariableop_21_adam_decayIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOp&assignvariableop_22_adam_learning_rateIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOp+assignvariableop_23_embedding_32_embeddingsIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOp+assignvariableop_24_embedding_30_embeddingsIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOp+assignvariableop_25_embedding_31_embeddingsIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOp+assignvariableop_26_embedding_35_embeddingsIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOp+assignvariableop_27_embedding_33_embeddingsIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOp+assignvariableop_28_embedding_34_embeddingsIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOp4assignvariableop_29_normalize_5_normalization_5_meanIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOp8assignvariableop_30_normalize_5_normalization_5_varianceIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_31?
AssignVariableOp_31AssignVariableOp5assignvariableop_31_normalize_5_normalization_5_countIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32?
AssignVariableOp_32AssignVariableOpassignvariableop_32_totalIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33?
AssignVariableOp_33AssignVariableOpassignvariableop_33_countIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34?
AssignVariableOp_34AssignVariableOp*assignvariableop_34_adam_dense_45_kernel_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35?
AssignVariableOp_35AssignVariableOp(assignvariableop_35_adam_dense_45_bias_mIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36?
AssignVariableOp_36AssignVariableOp*assignvariableop_36_adam_dense_46_kernel_mIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37?
AssignVariableOp_37AssignVariableOp(assignvariableop_37_adam_dense_46_bias_mIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38?
AssignVariableOp_38AssignVariableOp*assignvariableop_38_adam_dense_48_kernel_mIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39?
AssignVariableOp_39AssignVariableOp(assignvariableop_39_adam_dense_48_bias_mIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40?
AssignVariableOp_40AssignVariableOp*assignvariableop_40_adam_dense_47_kernel_mIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41?
AssignVariableOp_41AssignVariableOp(assignvariableop_41_adam_dense_47_bias_mIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42?
AssignVariableOp_42AssignVariableOp*assignvariableop_42_adam_dense_49_kernel_mIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43?
AssignVariableOp_43AssignVariableOp(assignvariableop_43_adam_dense_49_bias_mIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44?
AssignVariableOp_44AssignVariableOp*assignvariableop_44_adam_dense_50_kernel_mIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45?
AssignVariableOp_45AssignVariableOp(assignvariableop_45_adam_dense_50_bias_mIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46?
AssignVariableOp_46AssignVariableOp*assignvariableop_46_adam_dense_51_kernel_mIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47?
AssignVariableOp_47AssignVariableOp(assignvariableop_47_adam_dense_51_bias_mIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48?
AssignVariableOp_48AssignVariableOp*assignvariableop_48_adam_dense_52_kernel_mIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49?
AssignVariableOp_49AssignVariableOp(assignvariableop_49_adam_dense_52_bias_mIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50?
AssignVariableOp_50AssignVariableOp*assignvariableop_50_adam_dense_53_kernel_mIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51?
AssignVariableOp_51AssignVariableOp(assignvariableop_51_adam_dense_53_bias_mIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52?
AssignVariableOp_52AssignVariableOp2assignvariableop_52_adam_embedding_32_embeddings_mIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_52n
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:2
Identity_53?
AssignVariableOp_53AssignVariableOp2assignvariableop_53_adam_embedding_30_embeddings_mIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_53n
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:2
Identity_54?
AssignVariableOp_54AssignVariableOp2assignvariableop_54_adam_embedding_31_embeddings_mIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_54n
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:2
Identity_55?
AssignVariableOp_55AssignVariableOp2assignvariableop_55_adam_embedding_35_embeddings_mIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_55n
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:2
Identity_56?
AssignVariableOp_56AssignVariableOp2assignvariableop_56_adam_embedding_33_embeddings_mIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_56n
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:2
Identity_57?
AssignVariableOp_57AssignVariableOp2assignvariableop_57_adam_embedding_34_embeddings_mIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_57n
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:2
Identity_58?
AssignVariableOp_58AssignVariableOp*assignvariableop_58_adam_dense_45_kernel_vIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_58n
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:2
Identity_59?
AssignVariableOp_59AssignVariableOp(assignvariableop_59_adam_dense_45_bias_vIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_59n
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:2
Identity_60?
AssignVariableOp_60AssignVariableOp*assignvariableop_60_adam_dense_46_kernel_vIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_60n
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:2
Identity_61?
AssignVariableOp_61AssignVariableOp(assignvariableop_61_adam_dense_46_bias_vIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_61n
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:2
Identity_62?
AssignVariableOp_62AssignVariableOp*assignvariableop_62_adam_dense_48_kernel_vIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_62n
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:2
Identity_63?
AssignVariableOp_63AssignVariableOp(assignvariableop_63_adam_dense_48_bias_vIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_63n
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:2
Identity_64?
AssignVariableOp_64AssignVariableOp*assignvariableop_64_adam_dense_47_kernel_vIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_64n
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:2
Identity_65?
AssignVariableOp_65AssignVariableOp(assignvariableop_65_adam_dense_47_bias_vIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_65n
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:2
Identity_66?
AssignVariableOp_66AssignVariableOp*assignvariableop_66_adam_dense_49_kernel_vIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_66n
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:2
Identity_67?
AssignVariableOp_67AssignVariableOp(assignvariableop_67_adam_dense_49_bias_vIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_67n
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:2
Identity_68?
AssignVariableOp_68AssignVariableOp*assignvariableop_68_adam_dense_50_kernel_vIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_68n
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:2
Identity_69?
AssignVariableOp_69AssignVariableOp(assignvariableop_69_adam_dense_50_bias_vIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_69n
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:2
Identity_70?
AssignVariableOp_70AssignVariableOp*assignvariableop_70_adam_dense_51_kernel_vIdentity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_70n
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:2
Identity_71?
AssignVariableOp_71AssignVariableOp(assignvariableop_71_adam_dense_51_bias_vIdentity_71:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_71n
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:2
Identity_72?
AssignVariableOp_72AssignVariableOp*assignvariableop_72_adam_dense_52_kernel_vIdentity_72:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_72n
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:2
Identity_73?
AssignVariableOp_73AssignVariableOp(assignvariableop_73_adam_dense_52_bias_vIdentity_73:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_73n
Identity_74IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:2
Identity_74?
AssignVariableOp_74AssignVariableOp*assignvariableop_74_adam_dense_53_kernel_vIdentity_74:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_74n
Identity_75IdentityRestoreV2:tensors:75"/device:CPU:0*
T0*
_output_shapes
:2
Identity_75?
AssignVariableOp_75AssignVariableOp(assignvariableop_75_adam_dense_53_bias_vIdentity_75:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_75n
Identity_76IdentityRestoreV2:tensors:76"/device:CPU:0*
T0*
_output_shapes
:2
Identity_76?
AssignVariableOp_76AssignVariableOp2assignvariableop_76_adam_embedding_32_embeddings_vIdentity_76:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_76n
Identity_77IdentityRestoreV2:tensors:77"/device:CPU:0*
T0*
_output_shapes
:2
Identity_77?
AssignVariableOp_77AssignVariableOp2assignvariableop_77_adam_embedding_30_embeddings_vIdentity_77:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_77n
Identity_78IdentityRestoreV2:tensors:78"/device:CPU:0*
T0*
_output_shapes
:2
Identity_78?
AssignVariableOp_78AssignVariableOp2assignvariableop_78_adam_embedding_31_embeddings_vIdentity_78:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_78n
Identity_79IdentityRestoreV2:tensors:79"/device:CPU:0*
T0*
_output_shapes
:2
Identity_79?
AssignVariableOp_79AssignVariableOp2assignvariableop_79_adam_embedding_35_embeddings_vIdentity_79:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_79n
Identity_80IdentityRestoreV2:tensors:80"/device:CPU:0*
T0*
_output_shapes
:2
Identity_80?
AssignVariableOp_80AssignVariableOp2assignvariableop_80_adam_embedding_33_embeddings_vIdentity_80:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_80n
Identity_81IdentityRestoreV2:tensors:81"/device:CPU:0*
T0*
_output_shapes
:2
Identity_81?
AssignVariableOp_81AssignVariableOp2assignvariableop_81_adam_embedding_34_embeddings_vIdentity_81:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_819
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_82Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_82?
Identity_83IdentityIdentity_82:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_83"#
identity_83Identity_83:output:0*?
_input_shapes?
?: ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_66AssignVariableOp_662*
AssignVariableOp_67AssignVariableOp_672*
AssignVariableOp_68AssignVariableOp_682*
AssignVariableOp_69AssignVariableOp_692(
AssignVariableOp_7AssignVariableOp_72*
AssignVariableOp_70AssignVariableOp_702*
AssignVariableOp_71AssignVariableOp_712*
AssignVariableOp_72AssignVariableOp_722*
AssignVariableOp_73AssignVariableOp_732*
AssignVariableOp_74AssignVariableOp_742*
AssignVariableOp_75AssignVariableOp_752*
AssignVariableOp_76AssignVariableOp_762*
AssignVariableOp_77AssignVariableOp_772*
AssignVariableOp_78AssignVariableOp_782*
AssignVariableOp_79AssignVariableOp_792(
AssignVariableOp_8AssignVariableOp_82*
AssignVariableOp_80AssignVariableOp_802*
AssignVariableOp_81AssignVariableOp_812(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?	
?
J__inference_embedding_32_layer_call_and_return_conditional_losses_41025393

inputs
embedding_lookup_41025387
identity??embedding_lookup]
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:?????????2
Cast?
embedding_lookupResourceGatherembedding_lookup_41025387Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*,
_class"
 loc:@embedding_lookup/41025387*,
_output_shapes
:??????????*
dtype02
embedding_lookup?
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*,
_class"
 loc:@embedding_lookup/41025387*,
_output_shapes
:??????????2
embedding_lookup/Identity?
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????2
embedding_lookup/Identity_1?
IdentityIdentity$embedding_lookup/Identity_1:output:0^embedding_lookup*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????:2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
F__inference_dense_46_layer_call_and_return_conditional_losses_41027395

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?.
?
F__inference_model_11_layer_call_and_return_conditional_losses_41025796

inputs+
'tf_math_greater_equal_16_greaterequal_y
embedding_35_41025778
embedding_33_41025781
embedding_34_41025786
identity??$embedding_33/StatefulPartitionedCall?$embedding_34/StatefulPartitionedCall?$embedding_35/StatefulPartitionedCall?
flatten_11/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *Q
fLRJ
H__inference_flatten_11_layer_call_and_return_conditional_losses_410255912
flatten_11/PartitionedCall?
+tf.clip_by_value_16/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 * $tI2-
+tf.clip_by_value_16/clip_by_value/Minimum/y?
)tf.clip_by_value_16/clip_by_value/MinimumMinimum#flatten_11/PartitionedCall:output:04tf.clip_by_value_16/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2+
)tf.clip_by_value_16/clip_by_value/Minimum?
#tf.clip_by_value_16/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2%
#tf.clip_by_value_16/clip_by_value/y?
!tf.clip_by_value_16/clip_by_valueMaximum-tf.clip_by_value_16/clip_by_value/Minimum:z:0,tf.clip_by_value_16/clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????2#
!tf.clip_by_value_16/clip_by_value?
$tf.compat.v1.floor_div_11/FloorDiv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2&
$tf.compat.v1.floor_div_11/FloorDiv/y?
"tf.compat.v1.floor_div_11/FloorDivFloorDiv%tf.clip_by_value_16/clip_by_value:z:0-tf.compat.v1.floor_div_11/FloorDiv/y:output:0*
T0*'
_output_shapes
:?????????2$
"tf.compat.v1.floor_div_11/FloorDiv?
%tf.math.greater_equal_16/GreaterEqualGreaterEqual#flatten_11/PartitionedCall:output:0'tf_math_greater_equal_16_greaterequal_y*
T0*'
_output_shapes
:?????????2'
%tf.math.greater_equal_16/GreaterEqual?
tf.math.floormod_11/FloorMod/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2 
tf.math.floormod_11/FloorMod/y?
tf.math.floormod_11/FloorModFloorMod%tf.clip_by_value_16/clip_by_value:z:0'tf.math.floormod_11/FloorMod/y:output:0*
T0*'
_output_shapes
:?????????2
tf.math.floormod_11/FloorMod?
$embedding_35/StatefulPartitionedCallStatefulPartitionedCall%tf.clip_by_value_16/clip_by_value:z:0embedding_35_41025778*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *S
fNRL
J__inference_embedding_35_layer_call_and_return_conditional_losses_410256192&
$embedding_35/StatefulPartitionedCall?
$embedding_33/StatefulPartitionedCallStatefulPartitionedCall&tf.compat.v1.floor_div_11/FloorDiv:z:0embedding_33_41025781*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *S
fNRL
J__inference_embedding_33_layer_call_and_return_conditional_losses_410256412&
$embedding_33/StatefulPartitionedCall?
tf.cast_16/CastCast)tf.math.greater_equal_16/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2
tf.cast_16/Cast?
tf.__operators__.add_32/AddV2AddV2-embedding_35/StatefulPartitionedCall:output:0-embedding_33/StatefulPartitionedCall:output:0*
T0*,
_output_shapes
:??????????2
tf.__operators__.add_32/AddV2?
$embedding_34/StatefulPartitionedCallStatefulPartitionedCall tf.math.floormod_11/FloorMod:z:0embedding_34_41025786*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *S
fNRL
J__inference_embedding_34_layer_call_and_return_conditional_losses_410256652&
$embedding_34/StatefulPartitionedCall?
tf.__operators__.add_33/AddV2AddV2!tf.__operators__.add_32/AddV2:z:0-embedding_34/StatefulPartitionedCall:output:0*
T0*,
_output_shapes
:??????????2
tf.__operators__.add_33/AddV2?
 tf.expand_dims_11/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2"
 tf.expand_dims_11/ExpandDims/dim?
tf.expand_dims_11/ExpandDims
ExpandDimstf.cast_16/Cast:y:0)tf.expand_dims_11/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????2
tf.expand_dims_11/ExpandDims?
tf.math.multiply_11/MulMul!tf.__operators__.add_33/AddV2:z:0%tf.expand_dims_11/ExpandDims:output:0*
T0*,
_output_shapes
:??????????2
tf.math.multiply_11/Mul?
+tf.math.reduce_sum_11/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2-
+tf.math.reduce_sum_11/Sum/reduction_indices?
tf.math.reduce_sum_11/SumSumtf.math.multiply_11/Mul:z:04tf.math.reduce_sum_11/Sum/reduction_indices:output:0*
T0*(
_output_shapes
:??????????2
tf.math.reduce_sum_11/Sum?
IdentityIdentity"tf.math.reduce_sum_11/Sum:output:0%^embedding_33/StatefulPartitionedCall%^embedding_34/StatefulPartitionedCall%^embedding_35/StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:?????????: :::2L
$embedding_33/StatefulPartitionedCall$embedding_33/StatefulPartitionedCall2L
$embedding_34/StatefulPartitionedCall$embedding_34/StatefulPartitionedCall2L
$embedding_35/StatefulPartitionedCall$embedding_35/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: 
?	
?
F__inference_dense_49_layer_call_and_return_conditional_losses_41027453

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
J__inference_embedding_34_layer_call_and_return_conditional_losses_41025665

inputs
embedding_lookup_41025659
identity??embedding_lookup]
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:?????????2
Cast?
embedding_lookupResourceGatherembedding_lookup_41025659Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*,
_class"
 loc:@embedding_lookup/41025659*,
_output_shapes
:??????????*
dtype02
embedding_lookup?
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*,
_class"
 loc:@embedding_lookup/41025659*,
_output_shapes
:??????????2
embedding_lookup/Identity?
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????2
embedding_lookup/Identity_1?
IdentityIdentity$embedding_lookup/Identity_1:output:0^embedding_lookup*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????:2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
I__inference_normalize_5_layer_call_and_return_conditional_losses_41026129
x3
/normalization_5_reshape_readvariableop_resource5
1normalization_5_reshape_1_readvariableop_resource
identity??&normalization_5/Reshape/ReadVariableOp?(normalization_5/Reshape_1/ReadVariableOp?
&normalization_5/Reshape/ReadVariableOpReadVariableOp/normalization_5_reshape_readvariableop_resource*
_output_shapes	
:?*
dtype02(
&normalization_5/Reshape/ReadVariableOp?
normalization_5/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2
normalization_5/Reshape/shape?
normalization_5/ReshapeReshape.normalization_5/Reshape/ReadVariableOp:value:0&normalization_5/Reshape/shape:output:0*
T0*
_output_shapes
:	?2
normalization_5/Reshape?
(normalization_5/Reshape_1/ReadVariableOpReadVariableOp1normalization_5_reshape_1_readvariableop_resource*
_output_shapes	
:?*
dtype02*
(normalization_5/Reshape_1/ReadVariableOp?
normalization_5/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2!
normalization_5/Reshape_1/shape?
normalization_5/Reshape_1Reshape0normalization_5/Reshape_1/ReadVariableOp:value:0(normalization_5/Reshape_1/shape:output:0*
T0*
_output_shapes
:	?2
normalization_5/Reshape_1?
normalization_5/subSubx normalization_5/Reshape:output:0*
T0*(
_output_shapes
:??????????2
normalization_5/sub?
normalization_5/SqrtSqrt"normalization_5/Reshape_1:output:0*
T0*
_output_shapes
:	?2
normalization_5/Sqrt{
normalization_5/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32
normalization_5/Maximum/y?
normalization_5/MaximumMaximumnormalization_5/Sqrt:y:0"normalization_5/Maximum/y:output:0*
T0*
_output_shapes
:	?2
normalization_5/Maximum?
normalization_5/truedivRealDivnormalization_5/sub:z:0normalization_5/Maximum:z:0*
T0*(
_output_shapes
:??????????2
normalization_5/truediv?
IdentityIdentitynormalization_5/truediv:z:0'^normalization_5/Reshape/ReadVariableOp)^normalization_5/Reshape_1/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::2P
&normalization_5/Reshape/ReadVariableOp&normalization_5/Reshape/ReadVariableOp2T
(normalization_5/Reshape_1/ReadVariableOp(normalization_5/Reshape_1/ReadVariableOp:K G
(
_output_shapes
:??????????

_user_specified_namex
?	
?
F__inference_dense_51_layer_call_and_return_conditional_losses_41027491

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
+__inference_dense_52_layer_call_fn_41027519

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_dense_52_layer_call_and_return_conditional_losses_410260942
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
J__inference_embedding_33_layer_call_and_return_conditional_losses_41027664

inputs
embedding_lookup_41027658
identity??embedding_lookup]
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:?????????2
Cast?
embedding_lookupResourceGatherembedding_lookup_41027658Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*,
_class"
 loc:@embedding_lookup/41027658*,
_output_shapes
:??????????*
dtype02
embedding_lookup?
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*,
_class"
 loc:@embedding_lookup/41027658*,
_output_shapes
:??????????2
embedding_lookup/Identity?
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????2
embedding_lookup/Identity_1?
IdentityIdentity$embedding_lookup/Identity_1:output:0^embedding_lookup*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????:2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?Y
?

L__inference_custom_model_5_layer_call_and_return_conditional_losses_41026264

cards0

cards1
bets+
'tf_math_greater_equal_17_greaterequal_y
model_10_41026179
model_10_41026181
model_10_41026183
model_10_41026185
model_11_41026188
model_11_41026190
model_11_41026192
model_11_41026194/
+tf_clip_by_value_17_clip_by_value_minimum_y'
#tf_clip_by_value_17_clip_by_value_y
dense_45_41026206
dense_45_41026208
dense_48_41026211
dense_48_41026213
dense_46_41026216
dense_46_41026218
dense_47_41026221
dense_47_41026223
dense_49_41026226
dense_49_41026228
dense_50_41026233
dense_50_41026235
dense_51_41026239
dense_51_41026241
dense_52_41026246
dense_52_41026248
normalize_5_41026253
normalize_5_41026255
dense_53_41026258
dense_53_41026260
identity?? dense_45/StatefulPartitionedCall? dense_46/StatefulPartitionedCall? dense_47/StatefulPartitionedCall? dense_48/StatefulPartitionedCall? dense_49/StatefulPartitionedCall? dense_50/StatefulPartitionedCall? dense_51/StatefulPartitionedCall? dense_52/StatefulPartitionedCall? dense_53/StatefulPartitionedCall? model_10/StatefulPartitionedCall? model_11/StatefulPartitionedCall?#normalize_5/StatefulPartitionedCall?
%tf.math.greater_equal_17/GreaterEqualGreaterEqualbets'tf_math_greater_equal_17_greaterequal_y*
T0*'
_output_shapes
:?????????
2'
%tf.math.greater_equal_17/GreaterEqual?
 model_10/StatefulPartitionedCallStatefulPartitionedCallcards0model_10_41026179model_10_41026181model_10_41026183model_10_41026185*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*%
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_model_10_layer_call_and_return_conditional_losses_410255702"
 model_10/StatefulPartitionedCall?
 model_11/StatefulPartitionedCallStatefulPartitionedCallcards1model_11_41026188model_11_41026190model_11_41026192model_11_41026194*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*%
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_model_11_layer_call_and_return_conditional_losses_410257962"
 model_11/StatefulPartitionedCall?
)tf.clip_by_value_17/clip_by_value/MinimumMinimumbets+tf_clip_by_value_17_clip_by_value_minimum_y*
T0*'
_output_shapes
:?????????
2+
)tf.clip_by_value_17/clip_by_value/Minimum?
!tf.clip_by_value_17/clip_by_valueMaximum-tf.clip_by_value_17/clip_by_value/Minimum:z:0#tf_clip_by_value_17_clip_by_value_y*
T0*'
_output_shapes
:?????????
2#
!tf.clip_by_value_17/clip_by_value?
tf.cast_17/CastCast)tf.math.greater_equal_17/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????
2
tf.cast_17/Castv
tf.concat_15/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
tf.concat_15/concat/axis?
tf.concat_15/concatConcatV2)model_10/StatefulPartitionedCall:output:0)model_11/StatefulPartitionedCall:output:0!tf.concat_15/concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????2
tf.concat_15/concat
tf.concat_16/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
tf.concat_16/concat/axis?
tf.concat_16/concatConcatV2%tf.clip_by_value_17/clip_by_value:z:0tf.cast_17/Cast:y:0!tf.concat_16/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????2
tf.concat_16/concat?
 dense_45/StatefulPartitionedCallStatefulPartitionedCalltf.concat_15/concat:output:0dense_45_41026206dense_45_41026208*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_dense_45_layer_call_and_return_conditional_losses_410259052"
 dense_45/StatefulPartitionedCall?
 dense_48/StatefulPartitionedCallStatefulPartitionedCalltf.concat_16/concat:output:0dense_48_41026211dense_48_41026213*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_dense_48_layer_call_and_return_conditional_losses_410259312"
 dense_48/StatefulPartitionedCall?
 dense_46/StatefulPartitionedCallStatefulPartitionedCall)dense_45/StatefulPartitionedCall:output:0dense_46_41026216dense_46_41026218*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_dense_46_layer_call_and_return_conditional_losses_410259582"
 dense_46/StatefulPartitionedCall?
 dense_47/StatefulPartitionedCallStatefulPartitionedCall)dense_46/StatefulPartitionedCall:output:0dense_47_41026221dense_47_41026223*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_dense_47_layer_call_and_return_conditional_losses_410259852"
 dense_47/StatefulPartitionedCall?
 dense_49/StatefulPartitionedCallStatefulPartitionedCall)dense_48/StatefulPartitionedCall:output:0dense_49_41026226dense_49_41026228*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_dense_49_layer_call_and_return_conditional_losses_410260112"
 dense_49/StatefulPartitionedCall
tf.concat_17/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
tf.concat_17/concat/axis?
tf.concat_17/concatConcatV2)dense_47/StatefulPartitionedCall:output:0)dense_49/StatefulPartitionedCall:output:0!tf.concat_17/concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????2
tf.concat_17/concat?
 dense_50/StatefulPartitionedCallStatefulPartitionedCalltf.concat_17/concat:output:0dense_50_41026233dense_50_41026235*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_dense_50_layer_call_and_return_conditional_losses_410260392"
 dense_50/StatefulPartitionedCall?
tf.nn.relu_15/ReluRelu)dense_50/StatefulPartitionedCall:output:0*
T0*(
_output_shapes
:??????????2
tf.nn.relu_15/Relu?
 dense_51/StatefulPartitionedCallStatefulPartitionedCall tf.nn.relu_15/Relu:activations:0dense_51_41026239dense_51_41026241*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_dense_51_layer_call_and_return_conditional_losses_410260662"
 dense_51/StatefulPartitionedCall?
tf.__operators__.add_34/AddV2AddV2)dense_51/StatefulPartitionedCall:output:0 tf.nn.relu_15/Relu:activations:0*
T0*(
_output_shapes
:??????????2
tf.__operators__.add_34/AddV2?
tf.nn.relu_16/ReluRelu!tf.__operators__.add_34/AddV2:z:0*
T0*(
_output_shapes
:??????????2
tf.nn.relu_16/Relu?
 dense_52/StatefulPartitionedCallStatefulPartitionedCall tf.nn.relu_16/Relu:activations:0dense_52_41026246dense_52_41026248*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_dense_52_layer_call_and_return_conditional_losses_410260942"
 dense_52/StatefulPartitionedCall?
tf.__operators__.add_35/AddV2AddV2)dense_52/StatefulPartitionedCall:output:0 tf.nn.relu_16/Relu:activations:0*
T0*(
_output_shapes
:??????????2
tf.__operators__.add_35/AddV2?
tf.nn.relu_17/ReluRelu!tf.__operators__.add_35/AddV2:z:0*
T0*(
_output_shapes
:??????????2
tf.nn.relu_17/Relu?
#normalize_5/StatefulPartitionedCallStatefulPartitionedCall tf.nn.relu_17/Relu:activations:0normalize_5_41026253normalize_5_41026255*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *R
fMRK
I__inference_normalize_5_layer_call_and_return_conditional_losses_410261292%
#normalize_5/StatefulPartitionedCall?
 dense_53/StatefulPartitionedCallStatefulPartitionedCall,normalize_5/StatefulPartitionedCall:output:0dense_53_41026258dense_53_41026260*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_dense_53_layer_call_and_return_conditional_losses_410261552"
 dense_53/StatefulPartitionedCall?
IdentityIdentity)dense_53/StatefulPartitionedCall:output:0!^dense_45/StatefulPartitionedCall!^dense_46/StatefulPartitionedCall!^dense_47/StatefulPartitionedCall!^dense_48/StatefulPartitionedCall!^dense_49/StatefulPartitionedCall!^dense_50/StatefulPartitionedCall!^dense_51/StatefulPartitionedCall!^dense_52/StatefulPartitionedCall!^dense_53/StatefulPartitionedCall!^model_10/StatefulPartitionedCall!^model_11/StatefulPartitionedCall$^normalize_5/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????:?????????:?????????
: : :::: :::: : ::::::::::::::::::::2D
 dense_45/StatefulPartitionedCall dense_45/StatefulPartitionedCall2D
 dense_46/StatefulPartitionedCall dense_46/StatefulPartitionedCall2D
 dense_47/StatefulPartitionedCall dense_47/StatefulPartitionedCall2D
 dense_48/StatefulPartitionedCall dense_48/StatefulPartitionedCall2D
 dense_49/StatefulPartitionedCall dense_49/StatefulPartitionedCall2D
 dense_50/StatefulPartitionedCall dense_50/StatefulPartitionedCall2D
 dense_51/StatefulPartitionedCall dense_51/StatefulPartitionedCall2D
 dense_52/StatefulPartitionedCall dense_52/StatefulPartitionedCall2D
 dense_53/StatefulPartitionedCall dense_53/StatefulPartitionedCall2D
 model_10/StatefulPartitionedCall model_10/StatefulPartitionedCall2D
 model_11/StatefulPartitionedCall model_11/StatefulPartitionedCall2J
#normalize_5/StatefulPartitionedCall#normalize_5/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_namecards0:OK
'
_output_shapes
:?????????
 
_user_specified_namecards1:MI
'
_output_shapes
:?????????


_user_specified_namebets:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?9
?
F__inference_model_10_layer_call_and_return_conditional_losses_41027186

inputs+
'tf_math_greater_equal_15_greaterequal_y*
&embedding_32_embedding_lookup_41027160*
&embedding_30_embedding_lookup_41027166*
&embedding_31_embedding_lookup_41027174
identity??embedding_30/embedding_lookup?embedding_31/embedding_lookup?embedding_32/embedding_lookupu
flatten_10/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
flatten_10/Const?
flatten_10/ReshapeReshapeinputsflatten_10/Const:output:0*
T0*'
_output_shapes
:?????????2
flatten_10/Reshape?
+tf.clip_by_value_15/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 * $tI2-
+tf.clip_by_value_15/clip_by_value/Minimum/y?
)tf.clip_by_value_15/clip_by_value/MinimumMinimumflatten_10/Reshape:output:04tf.clip_by_value_15/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2+
)tf.clip_by_value_15/clip_by_value/Minimum?
#tf.clip_by_value_15/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2%
#tf.clip_by_value_15/clip_by_value/y?
!tf.clip_by_value_15/clip_by_valueMaximum-tf.clip_by_value_15/clip_by_value/Minimum:z:0,tf.clip_by_value_15/clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????2#
!tf.clip_by_value_15/clip_by_value?
$tf.compat.v1.floor_div_10/FloorDiv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2&
$tf.compat.v1.floor_div_10/FloorDiv/y?
"tf.compat.v1.floor_div_10/FloorDivFloorDiv%tf.clip_by_value_15/clip_by_value:z:0-tf.compat.v1.floor_div_10/FloorDiv/y:output:0*
T0*'
_output_shapes
:?????????2$
"tf.compat.v1.floor_div_10/FloorDiv?
%tf.math.greater_equal_15/GreaterEqualGreaterEqualflatten_10/Reshape:output:0'tf_math_greater_equal_15_greaterequal_y*
T0*'
_output_shapes
:?????????2'
%tf.math.greater_equal_15/GreaterEqual?
tf.math.floormod_10/FloorMod/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2 
tf.math.floormod_10/FloorMod/y?
tf.math.floormod_10/FloorModFloorMod%tf.clip_by_value_15/clip_by_value:z:0'tf.math.floormod_10/FloorMod/y:output:0*
T0*'
_output_shapes
:?????????2
tf.math.floormod_10/FloorMod?
embedding_32/CastCast%tf.clip_by_value_15/clip_by_value:z:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2
embedding_32/Cast?
embedding_32/embedding_lookupResourceGather&embedding_32_embedding_lookup_41027160embedding_32/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*9
_class/
-+loc:@embedding_32/embedding_lookup/41027160*,
_output_shapes
:??????????*
dtype02
embedding_32/embedding_lookup?
&embedding_32/embedding_lookup/IdentityIdentity&embedding_32/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*9
_class/
-+loc:@embedding_32/embedding_lookup/41027160*,
_output_shapes
:??????????2(
&embedding_32/embedding_lookup/Identity?
(embedding_32/embedding_lookup/Identity_1Identity/embedding_32/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????2*
(embedding_32/embedding_lookup/Identity_1?
embedding_30/CastCast&tf.compat.v1.floor_div_10/FloorDiv:z:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2
embedding_30/Cast?
embedding_30/embedding_lookupResourceGather&embedding_30_embedding_lookup_41027166embedding_30/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*9
_class/
-+loc:@embedding_30/embedding_lookup/41027166*,
_output_shapes
:??????????*
dtype02
embedding_30/embedding_lookup?
&embedding_30/embedding_lookup/IdentityIdentity&embedding_30/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*9
_class/
-+loc:@embedding_30/embedding_lookup/41027166*,
_output_shapes
:??????????2(
&embedding_30/embedding_lookup/Identity?
(embedding_30/embedding_lookup/Identity_1Identity/embedding_30/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????2*
(embedding_30/embedding_lookup/Identity_1?
tf.cast_15/CastCast)tf.math.greater_equal_15/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2
tf.cast_15/Cast?
tf.__operators__.add_30/AddV2AddV21embedding_32/embedding_lookup/Identity_1:output:01embedding_30/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:??????????2
tf.__operators__.add_30/AddV2?
embedding_31/CastCast tf.math.floormod_10/FloorMod:z:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2
embedding_31/Cast?
embedding_31/embedding_lookupResourceGather&embedding_31_embedding_lookup_41027174embedding_31/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*9
_class/
-+loc:@embedding_31/embedding_lookup/41027174*,
_output_shapes
:??????????*
dtype02
embedding_31/embedding_lookup?
&embedding_31/embedding_lookup/IdentityIdentity&embedding_31/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*9
_class/
-+loc:@embedding_31/embedding_lookup/41027174*,
_output_shapes
:??????????2(
&embedding_31/embedding_lookup/Identity?
(embedding_31/embedding_lookup/Identity_1Identity/embedding_31/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????2*
(embedding_31/embedding_lookup/Identity_1?
tf.__operators__.add_31/AddV2AddV2!tf.__operators__.add_30/AddV2:z:01embedding_31/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:??????????2
tf.__operators__.add_31/AddV2?
 tf.expand_dims_10/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2"
 tf.expand_dims_10/ExpandDims/dim?
tf.expand_dims_10/ExpandDims
ExpandDimstf.cast_15/Cast:y:0)tf.expand_dims_10/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????2
tf.expand_dims_10/ExpandDims?
tf.math.multiply_10/MulMul!tf.__operators__.add_31/AddV2:z:0%tf.expand_dims_10/ExpandDims:output:0*
T0*,
_output_shapes
:??????????2
tf.math.multiply_10/Mul?
+tf.math.reduce_sum_10/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2-
+tf.math.reduce_sum_10/Sum/reduction_indices?
tf.math.reduce_sum_10/SumSumtf.math.multiply_10/Mul:z:04tf.math.reduce_sum_10/Sum/reduction_indices:output:0*
T0*(
_output_shapes
:??????????2
tf.math.reduce_sum_10/Sum?
IdentityIdentity"tf.math.reduce_sum_10/Sum:output:0^embedding_30/embedding_lookup^embedding_31/embedding_lookup^embedding_32/embedding_lookup*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:?????????: :::2>
embedding_30/embedding_lookupembedding_30/embedding_lookup2>
embedding_31/embedding_lookupembedding_31/embedding_lookup2>
embedding_32/embedding_lookupembedding_32/embedding_lookup:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: 
?
?
+__inference_dense_49_layer_call_fn_41027462

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_dense_49_layer_call_and_return_conditional_losses_410260112
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
+__inference_model_11_layer_call_fn_41027364

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*%
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_model_11_layer_call_and_return_conditional_losses_410257962
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:?????????: :::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: 
?
d
H__inference_flatten_11_layer_call_and_return_conditional_losses_41025591

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
Constg
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:?????????2	
Reshaped
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
F__inference_dense_46_layer_call_and_return_conditional_losses_41025958

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?9
?
F__inference_model_11_layer_call_and_return_conditional_losses_41027338

inputs+
'tf_math_greater_equal_16_greaterequal_y*
&embedding_35_embedding_lookup_41027312*
&embedding_33_embedding_lookup_41027318*
&embedding_34_embedding_lookup_41027326
identity??embedding_33/embedding_lookup?embedding_34/embedding_lookup?embedding_35/embedding_lookupu
flatten_11/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
flatten_11/Const?
flatten_11/ReshapeReshapeinputsflatten_11/Const:output:0*
T0*'
_output_shapes
:?????????2
flatten_11/Reshape?
+tf.clip_by_value_16/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 * $tI2-
+tf.clip_by_value_16/clip_by_value/Minimum/y?
)tf.clip_by_value_16/clip_by_value/MinimumMinimumflatten_11/Reshape:output:04tf.clip_by_value_16/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2+
)tf.clip_by_value_16/clip_by_value/Minimum?
#tf.clip_by_value_16/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2%
#tf.clip_by_value_16/clip_by_value/y?
!tf.clip_by_value_16/clip_by_valueMaximum-tf.clip_by_value_16/clip_by_value/Minimum:z:0,tf.clip_by_value_16/clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????2#
!tf.clip_by_value_16/clip_by_value?
$tf.compat.v1.floor_div_11/FloorDiv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2&
$tf.compat.v1.floor_div_11/FloorDiv/y?
"tf.compat.v1.floor_div_11/FloorDivFloorDiv%tf.clip_by_value_16/clip_by_value:z:0-tf.compat.v1.floor_div_11/FloorDiv/y:output:0*
T0*'
_output_shapes
:?????????2$
"tf.compat.v1.floor_div_11/FloorDiv?
%tf.math.greater_equal_16/GreaterEqualGreaterEqualflatten_11/Reshape:output:0'tf_math_greater_equal_16_greaterequal_y*
T0*'
_output_shapes
:?????????2'
%tf.math.greater_equal_16/GreaterEqual?
tf.math.floormod_11/FloorMod/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2 
tf.math.floormod_11/FloorMod/y?
tf.math.floormod_11/FloorModFloorMod%tf.clip_by_value_16/clip_by_value:z:0'tf.math.floormod_11/FloorMod/y:output:0*
T0*'
_output_shapes
:?????????2
tf.math.floormod_11/FloorMod?
embedding_35/CastCast%tf.clip_by_value_16/clip_by_value:z:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2
embedding_35/Cast?
embedding_35/embedding_lookupResourceGather&embedding_35_embedding_lookup_41027312embedding_35/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*9
_class/
-+loc:@embedding_35/embedding_lookup/41027312*,
_output_shapes
:??????????*
dtype02
embedding_35/embedding_lookup?
&embedding_35/embedding_lookup/IdentityIdentity&embedding_35/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*9
_class/
-+loc:@embedding_35/embedding_lookup/41027312*,
_output_shapes
:??????????2(
&embedding_35/embedding_lookup/Identity?
(embedding_35/embedding_lookup/Identity_1Identity/embedding_35/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????2*
(embedding_35/embedding_lookup/Identity_1?
embedding_33/CastCast&tf.compat.v1.floor_div_11/FloorDiv:z:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2
embedding_33/Cast?
embedding_33/embedding_lookupResourceGather&embedding_33_embedding_lookup_41027318embedding_33/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*9
_class/
-+loc:@embedding_33/embedding_lookup/41027318*,
_output_shapes
:??????????*
dtype02
embedding_33/embedding_lookup?
&embedding_33/embedding_lookup/IdentityIdentity&embedding_33/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*9
_class/
-+loc:@embedding_33/embedding_lookup/41027318*,
_output_shapes
:??????????2(
&embedding_33/embedding_lookup/Identity?
(embedding_33/embedding_lookup/Identity_1Identity/embedding_33/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????2*
(embedding_33/embedding_lookup/Identity_1?
tf.cast_16/CastCast)tf.math.greater_equal_16/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2
tf.cast_16/Cast?
tf.__operators__.add_32/AddV2AddV21embedding_35/embedding_lookup/Identity_1:output:01embedding_33/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:??????????2
tf.__operators__.add_32/AddV2?
embedding_34/CastCast tf.math.floormod_11/FloorMod:z:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2
embedding_34/Cast?
embedding_34/embedding_lookupResourceGather&embedding_34_embedding_lookup_41027326embedding_34/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*9
_class/
-+loc:@embedding_34/embedding_lookup/41027326*,
_output_shapes
:??????????*
dtype02
embedding_34/embedding_lookup?
&embedding_34/embedding_lookup/IdentityIdentity&embedding_34/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*9
_class/
-+loc:@embedding_34/embedding_lookup/41027326*,
_output_shapes
:??????????2(
&embedding_34/embedding_lookup/Identity?
(embedding_34/embedding_lookup/Identity_1Identity/embedding_34/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????2*
(embedding_34/embedding_lookup/Identity_1?
tf.__operators__.add_33/AddV2AddV2!tf.__operators__.add_32/AddV2:z:01embedding_34/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:??????????2
tf.__operators__.add_33/AddV2?
 tf.expand_dims_11/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2"
 tf.expand_dims_11/ExpandDims/dim?
tf.expand_dims_11/ExpandDims
ExpandDimstf.cast_16/Cast:y:0)tf.expand_dims_11/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????2
tf.expand_dims_11/ExpandDims?
tf.math.multiply_11/MulMul!tf.__operators__.add_33/AddV2:z:0%tf.expand_dims_11/ExpandDims:output:0*
T0*,
_output_shapes
:??????????2
tf.math.multiply_11/Mul?
+tf.math.reduce_sum_11/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2-
+tf.math.reduce_sum_11/Sum/reduction_indices?
tf.math.reduce_sum_11/SumSumtf.math.multiply_11/Mul:z:04tf.math.reduce_sum_11/Sum/reduction_indices:output:0*
T0*(
_output_shapes
:??????????2
tf.math.reduce_sum_11/Sum?
IdentityIdentity"tf.math.reduce_sum_11/Sum:output:0^embedding_33/embedding_lookup^embedding_34/embedding_lookup^embedding_35/embedding_lookup*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:?????????: :::2>
embedding_33/embedding_lookupembedding_33/embedding_lookup2>
embedding_34/embedding_lookupembedding_34/embedding_lookup2>
embedding_35/embedding_lookupembedding_35/embedding_lookup:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: 
??
?#
!__inference__traced_save_41027964
file_prefix.
*savev2_dense_45_kernel_read_readvariableop,
(savev2_dense_45_bias_read_readvariableop.
*savev2_dense_46_kernel_read_readvariableop,
(savev2_dense_46_bias_read_readvariableop.
*savev2_dense_48_kernel_read_readvariableop,
(savev2_dense_48_bias_read_readvariableop.
*savev2_dense_47_kernel_read_readvariableop,
(savev2_dense_47_bias_read_readvariableop.
*savev2_dense_49_kernel_read_readvariableop,
(savev2_dense_49_bias_read_readvariableop.
*savev2_dense_50_kernel_read_readvariableop,
(savev2_dense_50_bias_read_readvariableop.
*savev2_dense_51_kernel_read_readvariableop,
(savev2_dense_51_bias_read_readvariableop.
*savev2_dense_52_kernel_read_readvariableop,
(savev2_dense_52_bias_read_readvariableop.
*savev2_dense_53_kernel_read_readvariableop,
(savev2_dense_53_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop6
2savev2_embedding_32_embeddings_read_readvariableop6
2savev2_embedding_30_embeddings_read_readvariableop6
2savev2_embedding_31_embeddings_read_readvariableop6
2savev2_embedding_35_embeddings_read_readvariableop6
2savev2_embedding_33_embeddings_read_readvariableop6
2savev2_embedding_34_embeddings_read_readvariableop?
;savev2_normalize_5_normalization_5_mean_read_readvariableopC
?savev2_normalize_5_normalization_5_variance_read_readvariableop@
<savev2_normalize_5_normalization_5_count_read_readvariableop	$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop5
1savev2_adam_dense_45_kernel_m_read_readvariableop3
/savev2_adam_dense_45_bias_m_read_readvariableop5
1savev2_adam_dense_46_kernel_m_read_readvariableop3
/savev2_adam_dense_46_bias_m_read_readvariableop5
1savev2_adam_dense_48_kernel_m_read_readvariableop3
/savev2_adam_dense_48_bias_m_read_readvariableop5
1savev2_adam_dense_47_kernel_m_read_readvariableop3
/savev2_adam_dense_47_bias_m_read_readvariableop5
1savev2_adam_dense_49_kernel_m_read_readvariableop3
/savev2_adam_dense_49_bias_m_read_readvariableop5
1savev2_adam_dense_50_kernel_m_read_readvariableop3
/savev2_adam_dense_50_bias_m_read_readvariableop5
1savev2_adam_dense_51_kernel_m_read_readvariableop3
/savev2_adam_dense_51_bias_m_read_readvariableop5
1savev2_adam_dense_52_kernel_m_read_readvariableop3
/savev2_adam_dense_52_bias_m_read_readvariableop5
1savev2_adam_dense_53_kernel_m_read_readvariableop3
/savev2_adam_dense_53_bias_m_read_readvariableop=
9savev2_adam_embedding_32_embeddings_m_read_readvariableop=
9savev2_adam_embedding_30_embeddings_m_read_readvariableop=
9savev2_adam_embedding_31_embeddings_m_read_readvariableop=
9savev2_adam_embedding_35_embeddings_m_read_readvariableop=
9savev2_adam_embedding_33_embeddings_m_read_readvariableop=
9savev2_adam_embedding_34_embeddings_m_read_readvariableop5
1savev2_adam_dense_45_kernel_v_read_readvariableop3
/savev2_adam_dense_45_bias_v_read_readvariableop5
1savev2_adam_dense_46_kernel_v_read_readvariableop3
/savev2_adam_dense_46_bias_v_read_readvariableop5
1savev2_adam_dense_48_kernel_v_read_readvariableop3
/savev2_adam_dense_48_bias_v_read_readvariableop5
1savev2_adam_dense_47_kernel_v_read_readvariableop3
/savev2_adam_dense_47_bias_v_read_readvariableop5
1savev2_adam_dense_49_kernel_v_read_readvariableop3
/savev2_adam_dense_49_bias_v_read_readvariableop5
1savev2_adam_dense_50_kernel_v_read_readvariableop3
/savev2_adam_dense_50_bias_v_read_readvariableop5
1savev2_adam_dense_51_kernel_v_read_readvariableop3
/savev2_adam_dense_51_bias_v_read_readvariableop5
1savev2_adam_dense_52_kernel_v_read_readvariableop3
/savev2_adam_dense_52_bias_v_read_readvariableop5
1savev2_adam_dense_53_kernel_v_read_readvariableop3
/savev2_adam_dense_53_bias_v_read_readvariableop=
9savev2_adam_embedding_32_embeddings_v_read_readvariableop=
9savev2_adam_embedding_30_embeddings_v_read_readvariableop=
9savev2_adam_embedding_31_embeddings_v_read_readvariableop=
9savev2_adam_embedding_35_embeddings_v_read_readvariableop=
9savev2_adam_embedding_33_embeddings_v_read_readvariableop=
9savev2_adam_embedding_34_embeddings_v_read_readvariableop
savev2_const_5

identity_1??MergeV2Checkpoints?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?-
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:S*
dtype0*?,
value?,B?,SB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:S*
dtype0*?
value?B?SB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?!
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_dense_45_kernel_read_readvariableop(savev2_dense_45_bias_read_readvariableop*savev2_dense_46_kernel_read_readvariableop(savev2_dense_46_bias_read_readvariableop*savev2_dense_48_kernel_read_readvariableop(savev2_dense_48_bias_read_readvariableop*savev2_dense_47_kernel_read_readvariableop(savev2_dense_47_bias_read_readvariableop*savev2_dense_49_kernel_read_readvariableop(savev2_dense_49_bias_read_readvariableop*savev2_dense_50_kernel_read_readvariableop(savev2_dense_50_bias_read_readvariableop*savev2_dense_51_kernel_read_readvariableop(savev2_dense_51_bias_read_readvariableop*savev2_dense_52_kernel_read_readvariableop(savev2_dense_52_bias_read_readvariableop*savev2_dense_53_kernel_read_readvariableop(savev2_dense_53_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop2savev2_embedding_32_embeddings_read_readvariableop2savev2_embedding_30_embeddings_read_readvariableop2savev2_embedding_31_embeddings_read_readvariableop2savev2_embedding_35_embeddings_read_readvariableop2savev2_embedding_33_embeddings_read_readvariableop2savev2_embedding_34_embeddings_read_readvariableop;savev2_normalize_5_normalization_5_mean_read_readvariableop?savev2_normalize_5_normalization_5_variance_read_readvariableop<savev2_normalize_5_normalization_5_count_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop1savev2_adam_dense_45_kernel_m_read_readvariableop/savev2_adam_dense_45_bias_m_read_readvariableop1savev2_adam_dense_46_kernel_m_read_readvariableop/savev2_adam_dense_46_bias_m_read_readvariableop1savev2_adam_dense_48_kernel_m_read_readvariableop/savev2_adam_dense_48_bias_m_read_readvariableop1savev2_adam_dense_47_kernel_m_read_readvariableop/savev2_adam_dense_47_bias_m_read_readvariableop1savev2_adam_dense_49_kernel_m_read_readvariableop/savev2_adam_dense_49_bias_m_read_readvariableop1savev2_adam_dense_50_kernel_m_read_readvariableop/savev2_adam_dense_50_bias_m_read_readvariableop1savev2_adam_dense_51_kernel_m_read_readvariableop/savev2_adam_dense_51_bias_m_read_readvariableop1savev2_adam_dense_52_kernel_m_read_readvariableop/savev2_adam_dense_52_bias_m_read_readvariableop1savev2_adam_dense_53_kernel_m_read_readvariableop/savev2_adam_dense_53_bias_m_read_readvariableop9savev2_adam_embedding_32_embeddings_m_read_readvariableop9savev2_adam_embedding_30_embeddings_m_read_readvariableop9savev2_adam_embedding_31_embeddings_m_read_readvariableop9savev2_adam_embedding_35_embeddings_m_read_readvariableop9savev2_adam_embedding_33_embeddings_m_read_readvariableop9savev2_adam_embedding_34_embeddings_m_read_readvariableop1savev2_adam_dense_45_kernel_v_read_readvariableop/savev2_adam_dense_45_bias_v_read_readvariableop1savev2_adam_dense_46_kernel_v_read_readvariableop/savev2_adam_dense_46_bias_v_read_readvariableop1savev2_adam_dense_48_kernel_v_read_readvariableop/savev2_adam_dense_48_bias_v_read_readvariableop1savev2_adam_dense_47_kernel_v_read_readvariableop/savev2_adam_dense_47_bias_v_read_readvariableop1savev2_adam_dense_49_kernel_v_read_readvariableop/savev2_adam_dense_49_bias_v_read_readvariableop1savev2_adam_dense_50_kernel_v_read_readvariableop/savev2_adam_dense_50_bias_v_read_readvariableop1savev2_adam_dense_51_kernel_v_read_readvariableop/savev2_adam_dense_51_bias_v_read_readvariableop1savev2_adam_dense_52_kernel_v_read_readvariableop/savev2_adam_dense_52_bias_v_read_readvariableop1savev2_adam_dense_53_kernel_v_read_readvariableop/savev2_adam_dense_53_bias_v_read_readvariableop9savev2_adam_embedding_32_embeddings_v_read_readvariableop9savev2_adam_embedding_30_embeddings_v_read_readvariableop9savev2_adam_embedding_31_embeddings_v_read_readvariableop9savev2_adam_embedding_35_embeddings_v_read_readvariableop9savev2_adam_embedding_33_embeddings_v_read_readvariableop9savev2_adam_embedding_34_embeddings_v_read_readvariableopsavev2_const_5"/device:CPU:0*
_output_shapes
 *a
dtypesW
U2S		2
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*?
_input_shapes?
?: :
??:?:
??:?:	?:?:
??:?:
??:?:
??:?:
??:?:
??:?:	?:: : : : : :	4?:	?:	?:	4?:	?:	?:?:?: : : :
??:?:
??:?:	?:?:
??:?:
??:?:
??:?:
??:?:
??:?:	?::	4?:	?:	?:	4?:	?:	?:
??:?:
??:?:	?:?:
??:?:
??:?:
??:?:
??:?:
??:?:	?::	4?:	?:	?:	4?:	?:	?: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:%!

_output_shapes
:	?:!

_output_shapes	
:?:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:&	"
 
_output_shapes
:
??:!


_output_shapes	
:?:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:%!

_output_shapes
:	?: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	4?:%!

_output_shapes
:	?:%!

_output_shapes
:	?:%!

_output_shapes
:	4?:%!

_output_shapes
:	?:%!

_output_shapes
:	?:!

_output_shapes	
:?:!

_output_shapes	
:?: 

_output_shapes
: :!

_output_shapes
: :"

_output_shapes
: :&#"
 
_output_shapes
:
??:!$

_output_shapes	
:?:&%"
 
_output_shapes
:
??:!&

_output_shapes	
:?:%'!

_output_shapes
:	?:!(

_output_shapes	
:?:&)"
 
_output_shapes
:
??:!*

_output_shapes	
:?:&+"
 
_output_shapes
:
??:!,

_output_shapes	
:?:&-"
 
_output_shapes
:
??:!.

_output_shapes	
:?:&/"
 
_output_shapes
:
??:!0

_output_shapes	
:?:&1"
 
_output_shapes
:
??:!2

_output_shapes	
:?:%3!

_output_shapes
:	?: 4

_output_shapes
::%5!

_output_shapes
:	4?:%6!

_output_shapes
:	?:%7!

_output_shapes
:	?:%8!

_output_shapes
:	4?:%9!

_output_shapes
:	?:%:!

_output_shapes
:	?:&;"
 
_output_shapes
:
??:!<

_output_shapes	
:?:&="
 
_output_shapes
:
??:!>

_output_shapes	
:?:%?!

_output_shapes
:	?:!@

_output_shapes	
:?:&A"
 
_output_shapes
:
??:!B

_output_shapes	
:?:&C"
 
_output_shapes
:
??:!D

_output_shapes	
:?:&E"
 
_output_shapes
:
??:!F

_output_shapes	
:?:&G"
 
_output_shapes
:
??:!H

_output_shapes	
:?:&I"
 
_output_shapes
:
??:!J

_output_shapes	
:?:%K!

_output_shapes
:	?: L

_output_shapes
::%M!

_output_shapes
:	4?:%N!

_output_shapes
:	?:%O!

_output_shapes
:	?:%P!

_output_shapes
:	4?:%Q!

_output_shapes
:	?:%R!

_output_shapes
:	?:S

_output_shapes
: 
?	
?
J__inference_embedding_31_layer_call_and_return_conditional_losses_41027619

inputs
embedding_lookup_41027613
identity??embedding_lookup]
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:?????????2
Cast?
embedding_lookupResourceGatherembedding_lookup_41027613Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*,
_class"
 loc:@embedding_lookup/41027613*,
_output_shapes
:??????????*
dtype02
embedding_lookup?
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*,
_class"
 loc:@embedding_lookup/41027613*,
_output_shapes
:??????????2
embedding_lookup/Identity?
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????2
embedding_lookup/Identity_1?
IdentityIdentity$embedding_lookup/Identity_1:output:0^embedding_lookup*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????:2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?.
?
F__inference_model_11_layer_call_and_return_conditional_losses_41025751

inputs+
'tf_math_greater_equal_16_greaterequal_y
embedding_35_41025733
embedding_33_41025736
embedding_34_41025741
identity??$embedding_33/StatefulPartitionedCall?$embedding_34/StatefulPartitionedCall?$embedding_35/StatefulPartitionedCall?
flatten_11/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *Q
fLRJ
H__inference_flatten_11_layer_call_and_return_conditional_losses_410255912
flatten_11/PartitionedCall?
+tf.clip_by_value_16/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 * $tI2-
+tf.clip_by_value_16/clip_by_value/Minimum/y?
)tf.clip_by_value_16/clip_by_value/MinimumMinimum#flatten_11/PartitionedCall:output:04tf.clip_by_value_16/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2+
)tf.clip_by_value_16/clip_by_value/Minimum?
#tf.clip_by_value_16/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2%
#tf.clip_by_value_16/clip_by_value/y?
!tf.clip_by_value_16/clip_by_valueMaximum-tf.clip_by_value_16/clip_by_value/Minimum:z:0,tf.clip_by_value_16/clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????2#
!tf.clip_by_value_16/clip_by_value?
$tf.compat.v1.floor_div_11/FloorDiv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2&
$tf.compat.v1.floor_div_11/FloorDiv/y?
"tf.compat.v1.floor_div_11/FloorDivFloorDiv%tf.clip_by_value_16/clip_by_value:z:0-tf.compat.v1.floor_div_11/FloorDiv/y:output:0*
T0*'
_output_shapes
:?????????2$
"tf.compat.v1.floor_div_11/FloorDiv?
%tf.math.greater_equal_16/GreaterEqualGreaterEqual#flatten_11/PartitionedCall:output:0'tf_math_greater_equal_16_greaterequal_y*
T0*'
_output_shapes
:?????????2'
%tf.math.greater_equal_16/GreaterEqual?
tf.math.floormod_11/FloorMod/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2 
tf.math.floormod_11/FloorMod/y?
tf.math.floormod_11/FloorModFloorMod%tf.clip_by_value_16/clip_by_value:z:0'tf.math.floormod_11/FloorMod/y:output:0*
T0*'
_output_shapes
:?????????2
tf.math.floormod_11/FloorMod?
$embedding_35/StatefulPartitionedCallStatefulPartitionedCall%tf.clip_by_value_16/clip_by_value:z:0embedding_35_41025733*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *S
fNRL
J__inference_embedding_35_layer_call_and_return_conditional_losses_410256192&
$embedding_35/StatefulPartitionedCall?
$embedding_33/StatefulPartitionedCallStatefulPartitionedCall&tf.compat.v1.floor_div_11/FloorDiv:z:0embedding_33_41025736*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *S
fNRL
J__inference_embedding_33_layer_call_and_return_conditional_losses_410256412&
$embedding_33/StatefulPartitionedCall?
tf.cast_16/CastCast)tf.math.greater_equal_16/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2
tf.cast_16/Cast?
tf.__operators__.add_32/AddV2AddV2-embedding_35/StatefulPartitionedCall:output:0-embedding_33/StatefulPartitionedCall:output:0*
T0*,
_output_shapes
:??????????2
tf.__operators__.add_32/AddV2?
$embedding_34/StatefulPartitionedCallStatefulPartitionedCall tf.math.floormod_11/FloorMod:z:0embedding_34_41025741*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *S
fNRL
J__inference_embedding_34_layer_call_and_return_conditional_losses_410256652&
$embedding_34/StatefulPartitionedCall?
tf.__operators__.add_33/AddV2AddV2!tf.__operators__.add_32/AddV2:z:0-embedding_34/StatefulPartitionedCall:output:0*
T0*,
_output_shapes
:??????????2
tf.__operators__.add_33/AddV2?
 tf.expand_dims_11/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2"
 tf.expand_dims_11/ExpandDims/dim?
tf.expand_dims_11/ExpandDims
ExpandDimstf.cast_16/Cast:y:0)tf.expand_dims_11/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????2
tf.expand_dims_11/ExpandDims?
tf.math.multiply_11/MulMul!tf.__operators__.add_33/AddV2:z:0%tf.expand_dims_11/ExpandDims:output:0*
T0*,
_output_shapes
:??????????2
tf.math.multiply_11/Mul?
+tf.math.reduce_sum_11/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2-
+tf.math.reduce_sum_11/Sum/reduction_indices?
tf.math.reduce_sum_11/SumSumtf.math.multiply_11/Mul:z:04tf.math.reduce_sum_11/Sum/reduction_indices:output:0*
T0*(
_output_shapes
:??????????2
tf.math.reduce_sum_11/Sum?
IdentityIdentity"tf.math.reduce_sum_11/Sum:output:0%^embedding_33/StatefulPartitionedCall%^embedding_34/StatefulPartitionedCall%^embedding_35/StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:?????????: :::2L
$embedding_33/StatefulPartitionedCall$embedding_33/StatefulPartitionedCall2L
$embedding_34/StatefulPartitionedCall$embedding_34/StatefulPartitionedCall2L
$embedding_35/StatefulPartitionedCall$embedding_35/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: 
?
?
+__inference_model_11_layer_call_fn_41027351

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*%
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_model_11_layer_call_and_return_conditional_losses_410257512
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:?????????: :::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: 
?	
?
F__inference_dense_48_layer_call_and_return_conditional_losses_41027414

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
+__inference_dense_48_layer_call_fn_41027423

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_dense_48_layer_call_and_return_conditional_losses_410259312
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
J__inference_embedding_30_layer_call_and_return_conditional_losses_41025415

inputs
embedding_lookup_41025409
identity??embedding_lookup]
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:?????????2
Cast?
embedding_lookupResourceGatherembedding_lookup_41025409Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*,
_class"
 loc:@embedding_lookup/41025409*,
_output_shapes
:??????????*
dtype02
embedding_lookup?
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*,
_class"
 loc:@embedding_lookup/41025409*,
_output_shapes
:??????????2
embedding_lookup/Identity?
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????2
embedding_lookup/Identity_1?
IdentityIdentity$embedding_lookup/Identity_1:output:0^embedding_lookup*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????:2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
d
H__inference_flatten_11_layer_call_and_return_conditional_losses_41027632

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
Constg
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:?????????2	
Reshaped
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
+__inference_dense_45_layer_call_fn_41027384

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_dense_45_layer_call_and_return_conditional_losses_410259052
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
1__inference_custom_model_5_layer_call_fn_41026587

cards0

cards1
bets
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallcards0cards1betsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29*-
Tin&
$2"*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*<
_read_only_resource_inputs
	
 !*2
config_proto" 

CPU

GPU2 *0J 8? *U
fPRN
L__inference_custom_model_5_layer_call_and_return_conditional_losses_410265222
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????:?????????:?????????
: : :::: :::: : ::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_namecards0:OK
'
_output_shapes
:?????????
 
_user_specified_namecards1:MI
'
_output_shapes
:?????????


_user_specified_namebets:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?.
?
F__inference_model_11_layer_call_and_return_conditional_losses_41025684
input_12+
'tf_math_greater_equal_16_greaterequal_y
embedding_35_41025628
embedding_33_41025650
embedding_34_41025674
identity??$embedding_33/StatefulPartitionedCall?$embedding_34/StatefulPartitionedCall?$embedding_35/StatefulPartitionedCall?
flatten_11/PartitionedCallPartitionedCallinput_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *Q
fLRJ
H__inference_flatten_11_layer_call_and_return_conditional_losses_410255912
flatten_11/PartitionedCall?
+tf.clip_by_value_16/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 * $tI2-
+tf.clip_by_value_16/clip_by_value/Minimum/y?
)tf.clip_by_value_16/clip_by_value/MinimumMinimum#flatten_11/PartitionedCall:output:04tf.clip_by_value_16/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2+
)tf.clip_by_value_16/clip_by_value/Minimum?
#tf.clip_by_value_16/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2%
#tf.clip_by_value_16/clip_by_value/y?
!tf.clip_by_value_16/clip_by_valueMaximum-tf.clip_by_value_16/clip_by_value/Minimum:z:0,tf.clip_by_value_16/clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????2#
!tf.clip_by_value_16/clip_by_value?
$tf.compat.v1.floor_div_11/FloorDiv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2&
$tf.compat.v1.floor_div_11/FloorDiv/y?
"tf.compat.v1.floor_div_11/FloorDivFloorDiv%tf.clip_by_value_16/clip_by_value:z:0-tf.compat.v1.floor_div_11/FloorDiv/y:output:0*
T0*'
_output_shapes
:?????????2$
"tf.compat.v1.floor_div_11/FloorDiv?
%tf.math.greater_equal_16/GreaterEqualGreaterEqual#flatten_11/PartitionedCall:output:0'tf_math_greater_equal_16_greaterequal_y*
T0*'
_output_shapes
:?????????2'
%tf.math.greater_equal_16/GreaterEqual?
tf.math.floormod_11/FloorMod/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2 
tf.math.floormod_11/FloorMod/y?
tf.math.floormod_11/FloorModFloorMod%tf.clip_by_value_16/clip_by_value:z:0'tf.math.floormod_11/FloorMod/y:output:0*
T0*'
_output_shapes
:?????????2
tf.math.floormod_11/FloorMod?
$embedding_35/StatefulPartitionedCallStatefulPartitionedCall%tf.clip_by_value_16/clip_by_value:z:0embedding_35_41025628*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *S
fNRL
J__inference_embedding_35_layer_call_and_return_conditional_losses_410256192&
$embedding_35/StatefulPartitionedCall?
$embedding_33/StatefulPartitionedCallStatefulPartitionedCall&tf.compat.v1.floor_div_11/FloorDiv:z:0embedding_33_41025650*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *S
fNRL
J__inference_embedding_33_layer_call_and_return_conditional_losses_410256412&
$embedding_33/StatefulPartitionedCall?
tf.cast_16/CastCast)tf.math.greater_equal_16/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2
tf.cast_16/Cast?
tf.__operators__.add_32/AddV2AddV2-embedding_35/StatefulPartitionedCall:output:0-embedding_33/StatefulPartitionedCall:output:0*
T0*,
_output_shapes
:??????????2
tf.__operators__.add_32/AddV2?
$embedding_34/StatefulPartitionedCallStatefulPartitionedCall tf.math.floormod_11/FloorMod:z:0embedding_34_41025674*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *S
fNRL
J__inference_embedding_34_layer_call_and_return_conditional_losses_410256652&
$embedding_34/StatefulPartitionedCall?
tf.__operators__.add_33/AddV2AddV2!tf.__operators__.add_32/AddV2:z:0-embedding_34/StatefulPartitionedCall:output:0*
T0*,
_output_shapes
:??????????2
tf.__operators__.add_33/AddV2?
 tf.expand_dims_11/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2"
 tf.expand_dims_11/ExpandDims/dim?
tf.expand_dims_11/ExpandDims
ExpandDimstf.cast_16/Cast:y:0)tf.expand_dims_11/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????2
tf.expand_dims_11/ExpandDims?
tf.math.multiply_11/MulMul!tf.__operators__.add_33/AddV2:z:0%tf.expand_dims_11/ExpandDims:output:0*
T0*,
_output_shapes
:??????????2
tf.math.multiply_11/Mul?
+tf.math.reduce_sum_11/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2-
+tf.math.reduce_sum_11/Sum/reduction_indices?
tf.math.reduce_sum_11/SumSumtf.math.multiply_11/Mul:z:04tf.math.reduce_sum_11/Sum/reduction_indices:output:0*
T0*(
_output_shapes
:??????????2
tf.math.reduce_sum_11/Sum?
IdentityIdentity"tf.math.reduce_sum_11/Sum:output:0%^embedding_33/StatefulPartitionedCall%^embedding_34/StatefulPartitionedCall%^embedding_35/StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:?????????: :::2L
$embedding_33/StatefulPartitionedCall$embedding_33/StatefulPartitionedCall2L
$embedding_34/StatefulPartitionedCall$embedding_34/StatefulPartitionedCall2L
$embedding_35/StatefulPartitionedCall$embedding_35/StatefulPartitionedCall:Q M
'
_output_shapes
:?????????
"
_user_specified_name
input_12:

_output_shapes
: 
?
?
+__inference_model_10_layer_call_fn_41025581
input_11
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_11unknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*%
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_model_10_layer_call_and_return_conditional_losses_410255702
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:?????????: :::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:?????????
"
_user_specified_name
input_11:

_output_shapes
: 
?
?
+__inference_dense_53_layer_call_fn_41027564

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_dense_53_layer_call_and_return_conditional_losses_410261552
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?Y
?

L__inference_custom_model_5_layer_call_and_return_conditional_losses_41026522

inputs
inputs_1
inputs_2+
'tf_math_greater_equal_17_greaterequal_y
model_10_41026437
model_10_41026439
model_10_41026441
model_10_41026443
model_11_41026446
model_11_41026448
model_11_41026450
model_11_41026452/
+tf_clip_by_value_17_clip_by_value_minimum_y'
#tf_clip_by_value_17_clip_by_value_y
dense_45_41026464
dense_45_41026466
dense_48_41026469
dense_48_41026471
dense_46_41026474
dense_46_41026476
dense_47_41026479
dense_47_41026481
dense_49_41026484
dense_49_41026486
dense_50_41026491
dense_50_41026493
dense_51_41026497
dense_51_41026499
dense_52_41026504
dense_52_41026506
normalize_5_41026511
normalize_5_41026513
dense_53_41026516
dense_53_41026518
identity?? dense_45/StatefulPartitionedCall? dense_46/StatefulPartitionedCall? dense_47/StatefulPartitionedCall? dense_48/StatefulPartitionedCall? dense_49/StatefulPartitionedCall? dense_50/StatefulPartitionedCall? dense_51/StatefulPartitionedCall? dense_52/StatefulPartitionedCall? dense_53/StatefulPartitionedCall? model_10/StatefulPartitionedCall? model_11/StatefulPartitionedCall?#normalize_5/StatefulPartitionedCall?
%tf.math.greater_equal_17/GreaterEqualGreaterEqualinputs_2'tf_math_greater_equal_17_greaterequal_y*
T0*'
_output_shapes
:?????????
2'
%tf.math.greater_equal_17/GreaterEqual?
 model_10/StatefulPartitionedCallStatefulPartitionedCallinputsmodel_10_41026437model_10_41026439model_10_41026441model_10_41026443*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*%
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_model_10_layer_call_and_return_conditional_losses_410255702"
 model_10/StatefulPartitionedCall?
 model_11/StatefulPartitionedCallStatefulPartitionedCallinputs_1model_11_41026446model_11_41026448model_11_41026450model_11_41026452*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*%
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_model_11_layer_call_and_return_conditional_losses_410257962"
 model_11/StatefulPartitionedCall?
)tf.clip_by_value_17/clip_by_value/MinimumMinimuminputs_2+tf_clip_by_value_17_clip_by_value_minimum_y*
T0*'
_output_shapes
:?????????
2+
)tf.clip_by_value_17/clip_by_value/Minimum?
!tf.clip_by_value_17/clip_by_valueMaximum-tf.clip_by_value_17/clip_by_value/Minimum:z:0#tf_clip_by_value_17_clip_by_value_y*
T0*'
_output_shapes
:?????????
2#
!tf.clip_by_value_17/clip_by_value?
tf.cast_17/CastCast)tf.math.greater_equal_17/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????
2
tf.cast_17/Castv
tf.concat_15/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
tf.concat_15/concat/axis?
tf.concat_15/concatConcatV2)model_10/StatefulPartitionedCall:output:0)model_11/StatefulPartitionedCall:output:0!tf.concat_15/concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????2
tf.concat_15/concat
tf.concat_16/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
tf.concat_16/concat/axis?
tf.concat_16/concatConcatV2%tf.clip_by_value_17/clip_by_value:z:0tf.cast_17/Cast:y:0!tf.concat_16/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????2
tf.concat_16/concat?
 dense_45/StatefulPartitionedCallStatefulPartitionedCalltf.concat_15/concat:output:0dense_45_41026464dense_45_41026466*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_dense_45_layer_call_and_return_conditional_losses_410259052"
 dense_45/StatefulPartitionedCall?
 dense_48/StatefulPartitionedCallStatefulPartitionedCalltf.concat_16/concat:output:0dense_48_41026469dense_48_41026471*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_dense_48_layer_call_and_return_conditional_losses_410259312"
 dense_48/StatefulPartitionedCall?
 dense_46/StatefulPartitionedCallStatefulPartitionedCall)dense_45/StatefulPartitionedCall:output:0dense_46_41026474dense_46_41026476*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_dense_46_layer_call_and_return_conditional_losses_410259582"
 dense_46/StatefulPartitionedCall?
 dense_47/StatefulPartitionedCallStatefulPartitionedCall)dense_46/StatefulPartitionedCall:output:0dense_47_41026479dense_47_41026481*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_dense_47_layer_call_and_return_conditional_losses_410259852"
 dense_47/StatefulPartitionedCall?
 dense_49/StatefulPartitionedCallStatefulPartitionedCall)dense_48/StatefulPartitionedCall:output:0dense_49_41026484dense_49_41026486*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_dense_49_layer_call_and_return_conditional_losses_410260112"
 dense_49/StatefulPartitionedCall
tf.concat_17/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
tf.concat_17/concat/axis?
tf.concat_17/concatConcatV2)dense_47/StatefulPartitionedCall:output:0)dense_49/StatefulPartitionedCall:output:0!tf.concat_17/concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????2
tf.concat_17/concat?
 dense_50/StatefulPartitionedCallStatefulPartitionedCalltf.concat_17/concat:output:0dense_50_41026491dense_50_41026493*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_dense_50_layer_call_and_return_conditional_losses_410260392"
 dense_50/StatefulPartitionedCall?
tf.nn.relu_15/ReluRelu)dense_50/StatefulPartitionedCall:output:0*
T0*(
_output_shapes
:??????????2
tf.nn.relu_15/Relu?
 dense_51/StatefulPartitionedCallStatefulPartitionedCall tf.nn.relu_15/Relu:activations:0dense_51_41026497dense_51_41026499*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_dense_51_layer_call_and_return_conditional_losses_410260662"
 dense_51/StatefulPartitionedCall?
tf.__operators__.add_34/AddV2AddV2)dense_51/StatefulPartitionedCall:output:0 tf.nn.relu_15/Relu:activations:0*
T0*(
_output_shapes
:??????????2
tf.__operators__.add_34/AddV2?
tf.nn.relu_16/ReluRelu!tf.__operators__.add_34/AddV2:z:0*
T0*(
_output_shapes
:??????????2
tf.nn.relu_16/Relu?
 dense_52/StatefulPartitionedCallStatefulPartitionedCall tf.nn.relu_16/Relu:activations:0dense_52_41026504dense_52_41026506*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_dense_52_layer_call_and_return_conditional_losses_410260942"
 dense_52/StatefulPartitionedCall?
tf.__operators__.add_35/AddV2AddV2)dense_52/StatefulPartitionedCall:output:0 tf.nn.relu_16/Relu:activations:0*
T0*(
_output_shapes
:??????????2
tf.__operators__.add_35/AddV2?
tf.nn.relu_17/ReluRelu!tf.__operators__.add_35/AddV2:z:0*
T0*(
_output_shapes
:??????????2
tf.nn.relu_17/Relu?
#normalize_5/StatefulPartitionedCallStatefulPartitionedCall tf.nn.relu_17/Relu:activations:0normalize_5_41026511normalize_5_41026513*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *R
fMRK
I__inference_normalize_5_layer_call_and_return_conditional_losses_410261292%
#normalize_5/StatefulPartitionedCall?
 dense_53/StatefulPartitionedCallStatefulPartitionedCall,normalize_5/StatefulPartitionedCall:output:0dense_53_41026516dense_53_41026518*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_dense_53_layer_call_and_return_conditional_losses_410261552"
 dense_53/StatefulPartitionedCall?
IdentityIdentity)dense_53/StatefulPartitionedCall:output:0!^dense_45/StatefulPartitionedCall!^dense_46/StatefulPartitionedCall!^dense_47/StatefulPartitionedCall!^dense_48/StatefulPartitionedCall!^dense_49/StatefulPartitionedCall!^dense_50/StatefulPartitionedCall!^dense_51/StatefulPartitionedCall!^dense_52/StatefulPartitionedCall!^dense_53/StatefulPartitionedCall!^model_10/StatefulPartitionedCall!^model_11/StatefulPartitionedCall$^normalize_5/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????:?????????:?????????
: : :::: :::: : ::::::::::::::::::::2D
 dense_45/StatefulPartitionedCall dense_45/StatefulPartitionedCall2D
 dense_46/StatefulPartitionedCall dense_46/StatefulPartitionedCall2D
 dense_47/StatefulPartitionedCall dense_47/StatefulPartitionedCall2D
 dense_48/StatefulPartitionedCall dense_48/StatefulPartitionedCall2D
 dense_49/StatefulPartitionedCall dense_49/StatefulPartitionedCall2D
 dense_50/StatefulPartitionedCall dense_50/StatefulPartitionedCall2D
 dense_51/StatefulPartitionedCall dense_51/StatefulPartitionedCall2D
 dense_52/StatefulPartitionedCall dense_52/StatefulPartitionedCall2D
 dense_53/StatefulPartitionedCall dense_53/StatefulPartitionedCall2D
 model_10/StatefulPartitionedCall model_10/StatefulPartitionedCall2D
 model_11/StatefulPartitionedCall model_11/StatefulPartitionedCall2J
#normalize_5/StatefulPartitionedCall#normalize_5/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????

 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?	
?
J__inference_embedding_32_layer_call_and_return_conditional_losses_41027585

inputs
embedding_lookup_41027579
identity??embedding_lookup]
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:?????????2
Cast?
embedding_lookupResourceGatherembedding_lookup_41027579Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*,
_class"
 loc:@embedding_lookup/41027579*,
_output_shapes
:??????????*
dtype02
embedding_lookup?
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*,
_class"
 loc:@embedding_lookup/41027579*,
_output_shapes
:??????????2
embedding_lookup/Identity?
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????2
embedding_lookup/Identity_1?
IdentityIdentity$embedding_lookup/Identity_1:output:0^embedding_lookup*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????:2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
d
H__inference_flatten_10_layer_call_and_return_conditional_losses_41025365

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
Constg
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:?????????2	
Reshaped
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
F__inference_dense_45_layer_call_and_return_conditional_losses_41025905

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
+__inference_model_11_layer_call_fn_41025762
input_12
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_12unknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*%
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_model_11_layer_call_and_return_conditional_losses_410257512
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:?????????: :::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:?????????
"
_user_specified_name
input_12:

_output_shapes
: 
?	
?
J__inference_embedding_35_layer_call_and_return_conditional_losses_41027647

inputs
embedding_lookup_41027641
identity??embedding_lookup]
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:?????????2
Cast?
embedding_lookupResourceGatherembedding_lookup_41027641Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*,
_class"
 loc:@embedding_lookup/41027641*,
_output_shapes
:??????????*
dtype02
embedding_lookup?
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*,
_class"
 loc:@embedding_lookup/41027641*,
_output_shapes
:??????????2
embedding_lookup/Identity?
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????2
embedding_lookup/Identity_1?
IdentityIdentity$embedding_lookup/Identity_1:output:0^embedding_lookup*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????:2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
F__inference_dense_49_layer_call_and_return_conditional_losses_41026011

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
u
/__inference_embedding_32_layer_call_fn_41027592

inputs
unknown
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *S
fNRL
J__inference_embedding_32_layer_call_and_return_conditional_losses_410253932
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????:22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
+__inference_dense_51_layer_call_fn_41027500

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_dense_51_layer_call_and_return_conditional_losses_410260662
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
J__inference_embedding_33_layer_call_and_return_conditional_losses_41025641

inputs
embedding_lookup_41025635
identity??embedding_lookup]
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:?????????2
Cast?
embedding_lookupResourceGatherembedding_lookup_41025635Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*,
_class"
 loc:@embedding_lookup/41025635*,
_output_shapes
:??????????*
dtype02
embedding_lookup?
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*,
_class"
 loc:@embedding_lookup/41025635*,
_output_shapes
:??????????2
embedding_lookup/Identity?
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????2
embedding_lookup/Identity_1?
IdentityIdentity$embedding_lookup/Identity_1:output:0^embedding_lookup*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????:2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
F__inference_dense_50_layer_call_and_return_conditional_losses_41027472

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
J__inference_embedding_31_layer_call_and_return_conditional_losses_41025439

inputs
embedding_lookup_41025433
identity??embedding_lookup]
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:?????????2
Cast?
embedding_lookupResourceGatherembedding_lookup_41025433Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*,
_class"
 loc:@embedding_lookup/41025433*,
_output_shapes
:??????????*
dtype02
embedding_lookup?
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*,
_class"
 loc:@embedding_lookup/41025433*,
_output_shapes
:??????????2
embedding_lookup/Identity?
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????2
embedding_lookup/Identity_1?
IdentityIdentity$embedding_lookup/Identity_1:output:0^embedding_lookup*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????:2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
+__inference_model_10_layer_call_fn_41027254

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*%
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_model_10_layer_call_and_return_conditional_losses_410255702
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:?????????: :::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: 
?
?
1__inference_custom_model_5_layer_call_fn_41026426

cards0

cards1
bets
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallcards0cards1betsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29*-
Tin&
$2"*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*<
_read_only_resource_inputs
	
 !*2
config_proto" 

CPU

GPU2 *0J 8? *U
fPRN
L__inference_custom_model_5_layer_call_and_return_conditional_losses_410263612
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????:?????????:?????????
: : :::: :::: : ::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_namecards0:OK
'
_output_shapes
:?????????
 
_user_specified_namecards1:MI
'
_output_shapes
:?????????


_user_specified_namebets:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
I
-__inference_flatten_10_layer_call_fn_41027575

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *Q
fLRJ
H__inference_flatten_10_layer_call_and_return_conditional_losses_410253652
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
+__inference_dense_50_layer_call_fn_41027481

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_dense_50_layer_call_and_return_conditional_losses_410260392
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
+__inference_model_11_layer_call_fn_41025807
input_12
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_12unknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*%
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_model_11_layer_call_and_return_conditional_losses_410257962
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:?????????: :::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:?????????
"
_user_specified_name
input_12:

_output_shapes
: 
?
d
H__inference_flatten_10_layer_call_and_return_conditional_losses_41027570

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
Constg
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:?????????2	
Reshaped
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
J__inference_embedding_30_layer_call_and_return_conditional_losses_41027602

inputs
embedding_lookup_41027596
identity??embedding_lookup]
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:?????????2
Cast?
embedding_lookupResourceGatherembedding_lookup_41027596Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*,
_class"
 loc:@embedding_lookup/41027596*,
_output_shapes
:??????????*
dtype02
embedding_lookup?
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*,
_class"
 loc:@embedding_lookup/41027596*,
_output_shapes
:??????????2
embedding_lookup/Identity?
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????2
embedding_lookup/Identity_1?
IdentityIdentity$embedding_lookup/Identity_1:output:0^embedding_lookup*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????:2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
F__inference_dense_50_layer_call_and_return_conditional_losses_41026039

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?.
?
F__inference_model_10_layer_call_and_return_conditional_losses_41025525

inputs+
'tf_math_greater_equal_15_greaterequal_y
embedding_32_41025507
embedding_30_41025510
embedding_31_41025515
identity??$embedding_30/StatefulPartitionedCall?$embedding_31/StatefulPartitionedCall?$embedding_32/StatefulPartitionedCall?
flatten_10/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *Q
fLRJ
H__inference_flatten_10_layer_call_and_return_conditional_losses_410253652
flatten_10/PartitionedCall?
+tf.clip_by_value_15/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 * $tI2-
+tf.clip_by_value_15/clip_by_value/Minimum/y?
)tf.clip_by_value_15/clip_by_value/MinimumMinimum#flatten_10/PartitionedCall:output:04tf.clip_by_value_15/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2+
)tf.clip_by_value_15/clip_by_value/Minimum?
#tf.clip_by_value_15/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2%
#tf.clip_by_value_15/clip_by_value/y?
!tf.clip_by_value_15/clip_by_valueMaximum-tf.clip_by_value_15/clip_by_value/Minimum:z:0,tf.clip_by_value_15/clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????2#
!tf.clip_by_value_15/clip_by_value?
$tf.compat.v1.floor_div_10/FloorDiv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2&
$tf.compat.v1.floor_div_10/FloorDiv/y?
"tf.compat.v1.floor_div_10/FloorDivFloorDiv%tf.clip_by_value_15/clip_by_value:z:0-tf.compat.v1.floor_div_10/FloorDiv/y:output:0*
T0*'
_output_shapes
:?????????2$
"tf.compat.v1.floor_div_10/FloorDiv?
%tf.math.greater_equal_15/GreaterEqualGreaterEqual#flatten_10/PartitionedCall:output:0'tf_math_greater_equal_15_greaterequal_y*
T0*'
_output_shapes
:?????????2'
%tf.math.greater_equal_15/GreaterEqual?
tf.math.floormod_10/FloorMod/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2 
tf.math.floormod_10/FloorMod/y?
tf.math.floormod_10/FloorModFloorMod%tf.clip_by_value_15/clip_by_value:z:0'tf.math.floormod_10/FloorMod/y:output:0*
T0*'
_output_shapes
:?????????2
tf.math.floormod_10/FloorMod?
$embedding_32/StatefulPartitionedCallStatefulPartitionedCall%tf.clip_by_value_15/clip_by_value:z:0embedding_32_41025507*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *S
fNRL
J__inference_embedding_32_layer_call_and_return_conditional_losses_410253932&
$embedding_32/StatefulPartitionedCall?
$embedding_30/StatefulPartitionedCallStatefulPartitionedCall&tf.compat.v1.floor_div_10/FloorDiv:z:0embedding_30_41025510*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *S
fNRL
J__inference_embedding_30_layer_call_and_return_conditional_losses_410254152&
$embedding_30/StatefulPartitionedCall?
tf.cast_15/CastCast)tf.math.greater_equal_15/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2
tf.cast_15/Cast?
tf.__operators__.add_30/AddV2AddV2-embedding_32/StatefulPartitionedCall:output:0-embedding_30/StatefulPartitionedCall:output:0*
T0*,
_output_shapes
:??????????2
tf.__operators__.add_30/AddV2?
$embedding_31/StatefulPartitionedCallStatefulPartitionedCall tf.math.floormod_10/FloorMod:z:0embedding_31_41025515*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *S
fNRL
J__inference_embedding_31_layer_call_and_return_conditional_losses_410254392&
$embedding_31/StatefulPartitionedCall?
tf.__operators__.add_31/AddV2AddV2!tf.__operators__.add_30/AddV2:z:0-embedding_31/StatefulPartitionedCall:output:0*
T0*,
_output_shapes
:??????????2
tf.__operators__.add_31/AddV2?
 tf.expand_dims_10/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2"
 tf.expand_dims_10/ExpandDims/dim?
tf.expand_dims_10/ExpandDims
ExpandDimstf.cast_15/Cast:y:0)tf.expand_dims_10/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????2
tf.expand_dims_10/ExpandDims?
tf.math.multiply_10/MulMul!tf.__operators__.add_31/AddV2:z:0%tf.expand_dims_10/ExpandDims:output:0*
T0*,
_output_shapes
:??????????2
tf.math.multiply_10/Mul?
+tf.math.reduce_sum_10/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2-
+tf.math.reduce_sum_10/Sum/reduction_indices?
tf.math.reduce_sum_10/SumSumtf.math.multiply_10/Mul:z:04tf.math.reduce_sum_10/Sum/reduction_indices:output:0*
T0*(
_output_shapes
:??????????2
tf.math.reduce_sum_10/Sum?
IdentityIdentity"tf.math.reduce_sum_10/Sum:output:0%^embedding_30/StatefulPartitionedCall%^embedding_31/StatefulPartitionedCall%^embedding_32/StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:?????????: :::2L
$embedding_30/StatefulPartitionedCall$embedding_30/StatefulPartitionedCall2L
$embedding_31/StatefulPartitionedCall$embedding_31/StatefulPartitionedCall2L
$embedding_32/StatefulPartitionedCall$embedding_32/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: 
?
I
-__inference_flatten_11_layer_call_fn_41027637

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *Q
fLRJ
H__inference_flatten_11_layer_call_and_return_conditional_losses_410255912
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
F__inference_dense_52_layer_call_and_return_conditional_losses_41026094

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?.
?
F__inference_model_10_layer_call_and_return_conditional_losses_41025490
input_11+
'tf_math_greater_equal_15_greaterequal_y
embedding_32_41025472
embedding_30_41025475
embedding_31_41025480
identity??$embedding_30/StatefulPartitionedCall?$embedding_31/StatefulPartitionedCall?$embedding_32/StatefulPartitionedCall?
flatten_10/PartitionedCallPartitionedCallinput_11*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *Q
fLRJ
H__inference_flatten_10_layer_call_and_return_conditional_losses_410253652
flatten_10/PartitionedCall?
+tf.clip_by_value_15/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 * $tI2-
+tf.clip_by_value_15/clip_by_value/Minimum/y?
)tf.clip_by_value_15/clip_by_value/MinimumMinimum#flatten_10/PartitionedCall:output:04tf.clip_by_value_15/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2+
)tf.clip_by_value_15/clip_by_value/Minimum?
#tf.clip_by_value_15/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2%
#tf.clip_by_value_15/clip_by_value/y?
!tf.clip_by_value_15/clip_by_valueMaximum-tf.clip_by_value_15/clip_by_value/Minimum:z:0,tf.clip_by_value_15/clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????2#
!tf.clip_by_value_15/clip_by_value?
$tf.compat.v1.floor_div_10/FloorDiv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2&
$tf.compat.v1.floor_div_10/FloorDiv/y?
"tf.compat.v1.floor_div_10/FloorDivFloorDiv%tf.clip_by_value_15/clip_by_value:z:0-tf.compat.v1.floor_div_10/FloorDiv/y:output:0*
T0*'
_output_shapes
:?????????2$
"tf.compat.v1.floor_div_10/FloorDiv?
%tf.math.greater_equal_15/GreaterEqualGreaterEqual#flatten_10/PartitionedCall:output:0'tf_math_greater_equal_15_greaterequal_y*
T0*'
_output_shapes
:?????????2'
%tf.math.greater_equal_15/GreaterEqual?
tf.math.floormod_10/FloorMod/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2 
tf.math.floormod_10/FloorMod/y?
tf.math.floormod_10/FloorModFloorMod%tf.clip_by_value_15/clip_by_value:z:0'tf.math.floormod_10/FloorMod/y:output:0*
T0*'
_output_shapes
:?????????2
tf.math.floormod_10/FloorMod?
$embedding_32/StatefulPartitionedCallStatefulPartitionedCall%tf.clip_by_value_15/clip_by_value:z:0embedding_32_41025472*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *S
fNRL
J__inference_embedding_32_layer_call_and_return_conditional_losses_410253932&
$embedding_32/StatefulPartitionedCall?
$embedding_30/StatefulPartitionedCallStatefulPartitionedCall&tf.compat.v1.floor_div_10/FloorDiv:z:0embedding_30_41025475*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *S
fNRL
J__inference_embedding_30_layer_call_and_return_conditional_losses_410254152&
$embedding_30/StatefulPartitionedCall?
tf.cast_15/CastCast)tf.math.greater_equal_15/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2
tf.cast_15/Cast?
tf.__operators__.add_30/AddV2AddV2-embedding_32/StatefulPartitionedCall:output:0-embedding_30/StatefulPartitionedCall:output:0*
T0*,
_output_shapes
:??????????2
tf.__operators__.add_30/AddV2?
$embedding_31/StatefulPartitionedCallStatefulPartitionedCall tf.math.floormod_10/FloorMod:z:0embedding_31_41025480*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *S
fNRL
J__inference_embedding_31_layer_call_and_return_conditional_losses_410254392&
$embedding_31/StatefulPartitionedCall?
tf.__operators__.add_31/AddV2AddV2!tf.__operators__.add_30/AddV2:z:0-embedding_31/StatefulPartitionedCall:output:0*
T0*,
_output_shapes
:??????????2
tf.__operators__.add_31/AddV2?
 tf.expand_dims_10/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2"
 tf.expand_dims_10/ExpandDims/dim?
tf.expand_dims_10/ExpandDims
ExpandDimstf.cast_15/Cast:y:0)tf.expand_dims_10/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????2
tf.expand_dims_10/ExpandDims?
tf.math.multiply_10/MulMul!tf.__operators__.add_31/AddV2:z:0%tf.expand_dims_10/ExpandDims:output:0*
T0*,
_output_shapes
:??????????2
tf.math.multiply_10/Mul?
+tf.math.reduce_sum_10/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2-
+tf.math.reduce_sum_10/Sum/reduction_indices?
tf.math.reduce_sum_10/SumSumtf.math.multiply_10/Mul:z:04tf.math.reduce_sum_10/Sum/reduction_indices:output:0*
T0*(
_output_shapes
:??????????2
tf.math.reduce_sum_10/Sum?
IdentityIdentity"tf.math.reduce_sum_10/Sum:output:0%^embedding_30/StatefulPartitionedCall%^embedding_31/StatefulPartitionedCall%^embedding_32/StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:?????????: :::2L
$embedding_30/StatefulPartitionedCall$embedding_30/StatefulPartitionedCall2L
$embedding_31/StatefulPartitionedCall$embedding_31/StatefulPartitionedCall2L
$embedding_32/StatefulPartitionedCall$embedding_32/StatefulPartitionedCall:Q M
'
_output_shapes
:?????????
"
_user_specified_name
input_11:

_output_shapes
: 
??
?
#__inference__wrapped_model_41025355

cards0

cards1
bets:
6custom_model_5_tf_math_greater_equal_17_greaterequal_yC
?custom_model_5_model_10_tf_math_greater_equal_15_greaterequal_yB
>custom_model_5_model_10_embedding_32_embedding_lookup_41025205B
>custom_model_5_model_10_embedding_30_embedding_lookup_41025211B
>custom_model_5_model_10_embedding_31_embedding_lookup_41025219C
?custom_model_5_model_11_tf_math_greater_equal_16_greaterequal_yB
>custom_model_5_model_11_embedding_35_embedding_lookup_41025243B
>custom_model_5_model_11_embedding_33_embedding_lookup_41025249B
>custom_model_5_model_11_embedding_34_embedding_lookup_41025257>
:custom_model_5_tf_clip_by_value_17_clip_by_value_minimum_y6
2custom_model_5_tf_clip_by_value_17_clip_by_value_y:
6custom_model_5_dense_45_matmul_readvariableop_resource;
7custom_model_5_dense_45_biasadd_readvariableop_resource:
6custom_model_5_dense_48_matmul_readvariableop_resource;
7custom_model_5_dense_48_biasadd_readvariableop_resource:
6custom_model_5_dense_46_matmul_readvariableop_resource;
7custom_model_5_dense_46_biasadd_readvariableop_resource:
6custom_model_5_dense_47_matmul_readvariableop_resource;
7custom_model_5_dense_47_biasadd_readvariableop_resource:
6custom_model_5_dense_49_matmul_readvariableop_resource;
7custom_model_5_dense_49_biasadd_readvariableop_resource:
6custom_model_5_dense_50_matmul_readvariableop_resource;
7custom_model_5_dense_50_biasadd_readvariableop_resource:
6custom_model_5_dense_51_matmul_readvariableop_resource;
7custom_model_5_dense_51_biasadd_readvariableop_resource:
6custom_model_5_dense_52_matmul_readvariableop_resource;
7custom_model_5_dense_52_biasadd_readvariableop_resourceN
Jcustom_model_5_normalize_5_normalization_5_reshape_readvariableop_resourceP
Lcustom_model_5_normalize_5_normalization_5_reshape_1_readvariableop_resource:
6custom_model_5_dense_53_matmul_readvariableop_resource;
7custom_model_5_dense_53_biasadd_readvariableop_resource
identity??.custom_model_5/dense_45/BiasAdd/ReadVariableOp?-custom_model_5/dense_45/MatMul/ReadVariableOp?.custom_model_5/dense_46/BiasAdd/ReadVariableOp?-custom_model_5/dense_46/MatMul/ReadVariableOp?.custom_model_5/dense_47/BiasAdd/ReadVariableOp?-custom_model_5/dense_47/MatMul/ReadVariableOp?.custom_model_5/dense_48/BiasAdd/ReadVariableOp?-custom_model_5/dense_48/MatMul/ReadVariableOp?.custom_model_5/dense_49/BiasAdd/ReadVariableOp?-custom_model_5/dense_49/MatMul/ReadVariableOp?.custom_model_5/dense_50/BiasAdd/ReadVariableOp?-custom_model_5/dense_50/MatMul/ReadVariableOp?.custom_model_5/dense_51/BiasAdd/ReadVariableOp?-custom_model_5/dense_51/MatMul/ReadVariableOp?.custom_model_5/dense_52/BiasAdd/ReadVariableOp?-custom_model_5/dense_52/MatMul/ReadVariableOp?.custom_model_5/dense_53/BiasAdd/ReadVariableOp?-custom_model_5/dense_53/MatMul/ReadVariableOp?5custom_model_5/model_10/embedding_30/embedding_lookup?5custom_model_5/model_10/embedding_31/embedding_lookup?5custom_model_5/model_10/embedding_32/embedding_lookup?5custom_model_5/model_11/embedding_33/embedding_lookup?5custom_model_5/model_11/embedding_34/embedding_lookup?5custom_model_5/model_11/embedding_35/embedding_lookup?Acustom_model_5/normalize_5/normalization_5/Reshape/ReadVariableOp?Ccustom_model_5/normalize_5/normalization_5/Reshape_1/ReadVariableOp?
4custom_model_5/tf.math.greater_equal_17/GreaterEqualGreaterEqualbets6custom_model_5_tf_math_greater_equal_17_greaterequal_y*
T0*'
_output_shapes
:?????????
26
4custom_model_5/tf.math.greater_equal_17/GreaterEqual?
(custom_model_5/model_10/flatten_10/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2*
(custom_model_5/model_10/flatten_10/Const?
*custom_model_5/model_10/flatten_10/ReshapeReshapecards01custom_model_5/model_10/flatten_10/Const:output:0*
T0*'
_output_shapes
:?????????2,
*custom_model_5/model_10/flatten_10/Reshape?
Ccustom_model_5/model_10/tf.clip_by_value_15/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 * $tI2E
Ccustom_model_5/model_10/tf.clip_by_value_15/clip_by_value/Minimum/y?
Acustom_model_5/model_10/tf.clip_by_value_15/clip_by_value/MinimumMinimum3custom_model_5/model_10/flatten_10/Reshape:output:0Lcustom_model_5/model_10/tf.clip_by_value_15/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2C
Acustom_model_5/model_10/tf.clip_by_value_15/clip_by_value/Minimum?
;custom_model_5/model_10/tf.clip_by_value_15/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2=
;custom_model_5/model_10/tf.clip_by_value_15/clip_by_value/y?
9custom_model_5/model_10/tf.clip_by_value_15/clip_by_valueMaximumEcustom_model_5/model_10/tf.clip_by_value_15/clip_by_value/Minimum:z:0Dcustom_model_5/model_10/tf.clip_by_value_15/clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????2;
9custom_model_5/model_10/tf.clip_by_value_15/clip_by_value?
<custom_model_5/model_10/tf.compat.v1.floor_div_10/FloorDiv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2>
<custom_model_5/model_10/tf.compat.v1.floor_div_10/FloorDiv/y?
:custom_model_5/model_10/tf.compat.v1.floor_div_10/FloorDivFloorDiv=custom_model_5/model_10/tf.clip_by_value_15/clip_by_value:z:0Ecustom_model_5/model_10/tf.compat.v1.floor_div_10/FloorDiv/y:output:0*
T0*'
_output_shapes
:?????????2<
:custom_model_5/model_10/tf.compat.v1.floor_div_10/FloorDiv?
=custom_model_5/model_10/tf.math.greater_equal_15/GreaterEqualGreaterEqual3custom_model_5/model_10/flatten_10/Reshape:output:0?custom_model_5_model_10_tf_math_greater_equal_15_greaterequal_y*
T0*'
_output_shapes
:?????????2?
=custom_model_5/model_10/tf.math.greater_equal_15/GreaterEqual?
6custom_model_5/model_10/tf.math.floormod_10/FloorMod/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@28
6custom_model_5/model_10/tf.math.floormod_10/FloorMod/y?
4custom_model_5/model_10/tf.math.floormod_10/FloorModFloorMod=custom_model_5/model_10/tf.clip_by_value_15/clip_by_value:z:0?custom_model_5/model_10/tf.math.floormod_10/FloorMod/y:output:0*
T0*'
_output_shapes
:?????????26
4custom_model_5/model_10/tf.math.floormod_10/FloorMod?
)custom_model_5/model_10/embedding_32/CastCast=custom_model_5/model_10/tf.clip_by_value_15/clip_by_value:z:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2+
)custom_model_5/model_10/embedding_32/Cast?
5custom_model_5/model_10/embedding_32/embedding_lookupResourceGather>custom_model_5_model_10_embedding_32_embedding_lookup_41025205-custom_model_5/model_10/embedding_32/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*Q
_classG
ECloc:@custom_model_5/model_10/embedding_32/embedding_lookup/41025205*,
_output_shapes
:??????????*
dtype027
5custom_model_5/model_10/embedding_32/embedding_lookup?
>custom_model_5/model_10/embedding_32/embedding_lookup/IdentityIdentity>custom_model_5/model_10/embedding_32/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*Q
_classG
ECloc:@custom_model_5/model_10/embedding_32/embedding_lookup/41025205*,
_output_shapes
:??????????2@
>custom_model_5/model_10/embedding_32/embedding_lookup/Identity?
@custom_model_5/model_10/embedding_32/embedding_lookup/Identity_1IdentityGcustom_model_5/model_10/embedding_32/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????2B
@custom_model_5/model_10/embedding_32/embedding_lookup/Identity_1?
)custom_model_5/model_10/embedding_30/CastCast>custom_model_5/model_10/tf.compat.v1.floor_div_10/FloorDiv:z:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2+
)custom_model_5/model_10/embedding_30/Cast?
5custom_model_5/model_10/embedding_30/embedding_lookupResourceGather>custom_model_5_model_10_embedding_30_embedding_lookup_41025211-custom_model_5/model_10/embedding_30/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*Q
_classG
ECloc:@custom_model_5/model_10/embedding_30/embedding_lookup/41025211*,
_output_shapes
:??????????*
dtype027
5custom_model_5/model_10/embedding_30/embedding_lookup?
>custom_model_5/model_10/embedding_30/embedding_lookup/IdentityIdentity>custom_model_5/model_10/embedding_30/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*Q
_classG
ECloc:@custom_model_5/model_10/embedding_30/embedding_lookup/41025211*,
_output_shapes
:??????????2@
>custom_model_5/model_10/embedding_30/embedding_lookup/Identity?
@custom_model_5/model_10/embedding_30/embedding_lookup/Identity_1IdentityGcustom_model_5/model_10/embedding_30/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????2B
@custom_model_5/model_10/embedding_30/embedding_lookup/Identity_1?
'custom_model_5/model_10/tf.cast_15/CastCastAcustom_model_5/model_10/tf.math.greater_equal_15/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2)
'custom_model_5/model_10/tf.cast_15/Cast?
5custom_model_5/model_10/tf.__operators__.add_30/AddV2AddV2Icustom_model_5/model_10/embedding_32/embedding_lookup/Identity_1:output:0Icustom_model_5/model_10/embedding_30/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:??????????27
5custom_model_5/model_10/tf.__operators__.add_30/AddV2?
)custom_model_5/model_10/embedding_31/CastCast8custom_model_5/model_10/tf.math.floormod_10/FloorMod:z:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2+
)custom_model_5/model_10/embedding_31/Cast?
5custom_model_5/model_10/embedding_31/embedding_lookupResourceGather>custom_model_5_model_10_embedding_31_embedding_lookup_41025219-custom_model_5/model_10/embedding_31/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*Q
_classG
ECloc:@custom_model_5/model_10/embedding_31/embedding_lookup/41025219*,
_output_shapes
:??????????*
dtype027
5custom_model_5/model_10/embedding_31/embedding_lookup?
>custom_model_5/model_10/embedding_31/embedding_lookup/IdentityIdentity>custom_model_5/model_10/embedding_31/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*Q
_classG
ECloc:@custom_model_5/model_10/embedding_31/embedding_lookup/41025219*,
_output_shapes
:??????????2@
>custom_model_5/model_10/embedding_31/embedding_lookup/Identity?
@custom_model_5/model_10/embedding_31/embedding_lookup/Identity_1IdentityGcustom_model_5/model_10/embedding_31/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????2B
@custom_model_5/model_10/embedding_31/embedding_lookup/Identity_1?
5custom_model_5/model_10/tf.__operators__.add_31/AddV2AddV29custom_model_5/model_10/tf.__operators__.add_30/AddV2:z:0Icustom_model_5/model_10/embedding_31/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:??????????27
5custom_model_5/model_10/tf.__operators__.add_31/AddV2?
8custom_model_5/model_10/tf.expand_dims_10/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2:
8custom_model_5/model_10/tf.expand_dims_10/ExpandDims/dim?
4custom_model_5/model_10/tf.expand_dims_10/ExpandDims
ExpandDims+custom_model_5/model_10/tf.cast_15/Cast:y:0Acustom_model_5/model_10/tf.expand_dims_10/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????26
4custom_model_5/model_10/tf.expand_dims_10/ExpandDims?
/custom_model_5/model_10/tf.math.multiply_10/MulMul9custom_model_5/model_10/tf.__operators__.add_31/AddV2:z:0=custom_model_5/model_10/tf.expand_dims_10/ExpandDims:output:0*
T0*,
_output_shapes
:??????????21
/custom_model_5/model_10/tf.math.multiply_10/Mul?
Ccustom_model_5/model_10/tf.math.reduce_sum_10/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2E
Ccustom_model_5/model_10/tf.math.reduce_sum_10/Sum/reduction_indices?
1custom_model_5/model_10/tf.math.reduce_sum_10/SumSum3custom_model_5/model_10/tf.math.multiply_10/Mul:z:0Lcustom_model_5/model_10/tf.math.reduce_sum_10/Sum/reduction_indices:output:0*
T0*(
_output_shapes
:??????????23
1custom_model_5/model_10/tf.math.reduce_sum_10/Sum?
(custom_model_5/model_11/flatten_11/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2*
(custom_model_5/model_11/flatten_11/Const?
*custom_model_5/model_11/flatten_11/ReshapeReshapecards11custom_model_5/model_11/flatten_11/Const:output:0*
T0*'
_output_shapes
:?????????2,
*custom_model_5/model_11/flatten_11/Reshape?
Ccustom_model_5/model_11/tf.clip_by_value_16/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 * $tI2E
Ccustom_model_5/model_11/tf.clip_by_value_16/clip_by_value/Minimum/y?
Acustom_model_5/model_11/tf.clip_by_value_16/clip_by_value/MinimumMinimum3custom_model_5/model_11/flatten_11/Reshape:output:0Lcustom_model_5/model_11/tf.clip_by_value_16/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2C
Acustom_model_5/model_11/tf.clip_by_value_16/clip_by_value/Minimum?
;custom_model_5/model_11/tf.clip_by_value_16/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2=
;custom_model_5/model_11/tf.clip_by_value_16/clip_by_value/y?
9custom_model_5/model_11/tf.clip_by_value_16/clip_by_valueMaximumEcustom_model_5/model_11/tf.clip_by_value_16/clip_by_value/Minimum:z:0Dcustom_model_5/model_11/tf.clip_by_value_16/clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????2;
9custom_model_5/model_11/tf.clip_by_value_16/clip_by_value?
<custom_model_5/model_11/tf.compat.v1.floor_div_11/FloorDiv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2>
<custom_model_5/model_11/tf.compat.v1.floor_div_11/FloorDiv/y?
:custom_model_5/model_11/tf.compat.v1.floor_div_11/FloorDivFloorDiv=custom_model_5/model_11/tf.clip_by_value_16/clip_by_value:z:0Ecustom_model_5/model_11/tf.compat.v1.floor_div_11/FloorDiv/y:output:0*
T0*'
_output_shapes
:?????????2<
:custom_model_5/model_11/tf.compat.v1.floor_div_11/FloorDiv?
=custom_model_5/model_11/tf.math.greater_equal_16/GreaterEqualGreaterEqual3custom_model_5/model_11/flatten_11/Reshape:output:0?custom_model_5_model_11_tf_math_greater_equal_16_greaterequal_y*
T0*'
_output_shapes
:?????????2?
=custom_model_5/model_11/tf.math.greater_equal_16/GreaterEqual?
6custom_model_5/model_11/tf.math.floormod_11/FloorMod/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@28
6custom_model_5/model_11/tf.math.floormod_11/FloorMod/y?
4custom_model_5/model_11/tf.math.floormod_11/FloorModFloorMod=custom_model_5/model_11/tf.clip_by_value_16/clip_by_value:z:0?custom_model_5/model_11/tf.math.floormod_11/FloorMod/y:output:0*
T0*'
_output_shapes
:?????????26
4custom_model_5/model_11/tf.math.floormod_11/FloorMod?
)custom_model_5/model_11/embedding_35/CastCast=custom_model_5/model_11/tf.clip_by_value_16/clip_by_value:z:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2+
)custom_model_5/model_11/embedding_35/Cast?
5custom_model_5/model_11/embedding_35/embedding_lookupResourceGather>custom_model_5_model_11_embedding_35_embedding_lookup_41025243-custom_model_5/model_11/embedding_35/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*Q
_classG
ECloc:@custom_model_5/model_11/embedding_35/embedding_lookup/41025243*,
_output_shapes
:??????????*
dtype027
5custom_model_5/model_11/embedding_35/embedding_lookup?
>custom_model_5/model_11/embedding_35/embedding_lookup/IdentityIdentity>custom_model_5/model_11/embedding_35/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*Q
_classG
ECloc:@custom_model_5/model_11/embedding_35/embedding_lookup/41025243*,
_output_shapes
:??????????2@
>custom_model_5/model_11/embedding_35/embedding_lookup/Identity?
@custom_model_5/model_11/embedding_35/embedding_lookup/Identity_1IdentityGcustom_model_5/model_11/embedding_35/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????2B
@custom_model_5/model_11/embedding_35/embedding_lookup/Identity_1?
)custom_model_5/model_11/embedding_33/CastCast>custom_model_5/model_11/tf.compat.v1.floor_div_11/FloorDiv:z:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2+
)custom_model_5/model_11/embedding_33/Cast?
5custom_model_5/model_11/embedding_33/embedding_lookupResourceGather>custom_model_5_model_11_embedding_33_embedding_lookup_41025249-custom_model_5/model_11/embedding_33/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*Q
_classG
ECloc:@custom_model_5/model_11/embedding_33/embedding_lookup/41025249*,
_output_shapes
:??????????*
dtype027
5custom_model_5/model_11/embedding_33/embedding_lookup?
>custom_model_5/model_11/embedding_33/embedding_lookup/IdentityIdentity>custom_model_5/model_11/embedding_33/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*Q
_classG
ECloc:@custom_model_5/model_11/embedding_33/embedding_lookup/41025249*,
_output_shapes
:??????????2@
>custom_model_5/model_11/embedding_33/embedding_lookup/Identity?
@custom_model_5/model_11/embedding_33/embedding_lookup/Identity_1IdentityGcustom_model_5/model_11/embedding_33/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????2B
@custom_model_5/model_11/embedding_33/embedding_lookup/Identity_1?
'custom_model_5/model_11/tf.cast_16/CastCastAcustom_model_5/model_11/tf.math.greater_equal_16/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2)
'custom_model_5/model_11/tf.cast_16/Cast?
5custom_model_5/model_11/tf.__operators__.add_32/AddV2AddV2Icustom_model_5/model_11/embedding_35/embedding_lookup/Identity_1:output:0Icustom_model_5/model_11/embedding_33/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:??????????27
5custom_model_5/model_11/tf.__operators__.add_32/AddV2?
)custom_model_5/model_11/embedding_34/CastCast8custom_model_5/model_11/tf.math.floormod_11/FloorMod:z:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2+
)custom_model_5/model_11/embedding_34/Cast?
5custom_model_5/model_11/embedding_34/embedding_lookupResourceGather>custom_model_5_model_11_embedding_34_embedding_lookup_41025257-custom_model_5/model_11/embedding_34/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*Q
_classG
ECloc:@custom_model_5/model_11/embedding_34/embedding_lookup/41025257*,
_output_shapes
:??????????*
dtype027
5custom_model_5/model_11/embedding_34/embedding_lookup?
>custom_model_5/model_11/embedding_34/embedding_lookup/IdentityIdentity>custom_model_5/model_11/embedding_34/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*Q
_classG
ECloc:@custom_model_5/model_11/embedding_34/embedding_lookup/41025257*,
_output_shapes
:??????????2@
>custom_model_5/model_11/embedding_34/embedding_lookup/Identity?
@custom_model_5/model_11/embedding_34/embedding_lookup/Identity_1IdentityGcustom_model_5/model_11/embedding_34/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????2B
@custom_model_5/model_11/embedding_34/embedding_lookup/Identity_1?
5custom_model_5/model_11/tf.__operators__.add_33/AddV2AddV29custom_model_5/model_11/tf.__operators__.add_32/AddV2:z:0Icustom_model_5/model_11/embedding_34/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:??????????27
5custom_model_5/model_11/tf.__operators__.add_33/AddV2?
8custom_model_5/model_11/tf.expand_dims_11/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2:
8custom_model_5/model_11/tf.expand_dims_11/ExpandDims/dim?
4custom_model_5/model_11/tf.expand_dims_11/ExpandDims
ExpandDims+custom_model_5/model_11/tf.cast_16/Cast:y:0Acustom_model_5/model_11/tf.expand_dims_11/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????26
4custom_model_5/model_11/tf.expand_dims_11/ExpandDims?
/custom_model_5/model_11/tf.math.multiply_11/MulMul9custom_model_5/model_11/tf.__operators__.add_33/AddV2:z:0=custom_model_5/model_11/tf.expand_dims_11/ExpandDims:output:0*
T0*,
_output_shapes
:??????????21
/custom_model_5/model_11/tf.math.multiply_11/Mul?
Ccustom_model_5/model_11/tf.math.reduce_sum_11/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2E
Ccustom_model_5/model_11/tf.math.reduce_sum_11/Sum/reduction_indices?
1custom_model_5/model_11/tf.math.reduce_sum_11/SumSum3custom_model_5/model_11/tf.math.multiply_11/Mul:z:0Lcustom_model_5/model_11/tf.math.reduce_sum_11/Sum/reduction_indices:output:0*
T0*(
_output_shapes
:??????????23
1custom_model_5/model_11/tf.math.reduce_sum_11/Sum?
8custom_model_5/tf.clip_by_value_17/clip_by_value/MinimumMinimumbets:custom_model_5_tf_clip_by_value_17_clip_by_value_minimum_y*
T0*'
_output_shapes
:?????????
2:
8custom_model_5/tf.clip_by_value_17/clip_by_value/Minimum?
0custom_model_5/tf.clip_by_value_17/clip_by_valueMaximum<custom_model_5/tf.clip_by_value_17/clip_by_value/Minimum:z:02custom_model_5_tf_clip_by_value_17_clip_by_value_y*
T0*'
_output_shapes
:?????????
22
0custom_model_5/tf.clip_by_value_17/clip_by_value?
custom_model_5/tf.cast_17/CastCast8custom_model_5/tf.math.greater_equal_17/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????
2 
custom_model_5/tf.cast_17/Cast?
'custom_model_5/tf.concat_15/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2)
'custom_model_5/tf.concat_15/concat/axis?
"custom_model_5/tf.concat_15/concatConcatV2:custom_model_5/model_10/tf.math.reduce_sum_10/Sum:output:0:custom_model_5/model_11/tf.math.reduce_sum_11/Sum:output:00custom_model_5/tf.concat_15/concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????2$
"custom_model_5/tf.concat_15/concat?
'custom_model_5/tf.concat_16/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2)
'custom_model_5/tf.concat_16/concat/axis?
"custom_model_5/tf.concat_16/concatConcatV24custom_model_5/tf.clip_by_value_17/clip_by_value:z:0"custom_model_5/tf.cast_17/Cast:y:00custom_model_5/tf.concat_16/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????2$
"custom_model_5/tf.concat_16/concat?
-custom_model_5/dense_45/MatMul/ReadVariableOpReadVariableOp6custom_model_5_dense_45_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02/
-custom_model_5/dense_45/MatMul/ReadVariableOp?
custom_model_5/dense_45/MatMulMatMul+custom_model_5/tf.concat_15/concat:output:05custom_model_5/dense_45/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2 
custom_model_5/dense_45/MatMul?
.custom_model_5/dense_45/BiasAdd/ReadVariableOpReadVariableOp7custom_model_5_dense_45_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype020
.custom_model_5/dense_45/BiasAdd/ReadVariableOp?
custom_model_5/dense_45/BiasAddBiasAdd(custom_model_5/dense_45/MatMul:product:06custom_model_5/dense_45/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2!
custom_model_5/dense_45/BiasAdd?
custom_model_5/dense_45/ReluRelu(custom_model_5/dense_45/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
custom_model_5/dense_45/Relu?
-custom_model_5/dense_48/MatMul/ReadVariableOpReadVariableOp6custom_model_5_dense_48_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02/
-custom_model_5/dense_48/MatMul/ReadVariableOp?
custom_model_5/dense_48/MatMulMatMul+custom_model_5/tf.concat_16/concat:output:05custom_model_5/dense_48/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2 
custom_model_5/dense_48/MatMul?
.custom_model_5/dense_48/BiasAdd/ReadVariableOpReadVariableOp7custom_model_5_dense_48_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype020
.custom_model_5/dense_48/BiasAdd/ReadVariableOp?
custom_model_5/dense_48/BiasAddBiasAdd(custom_model_5/dense_48/MatMul:product:06custom_model_5/dense_48/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2!
custom_model_5/dense_48/BiasAdd?
-custom_model_5/dense_46/MatMul/ReadVariableOpReadVariableOp6custom_model_5_dense_46_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02/
-custom_model_5/dense_46/MatMul/ReadVariableOp?
custom_model_5/dense_46/MatMulMatMul*custom_model_5/dense_45/Relu:activations:05custom_model_5/dense_46/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2 
custom_model_5/dense_46/MatMul?
.custom_model_5/dense_46/BiasAdd/ReadVariableOpReadVariableOp7custom_model_5_dense_46_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype020
.custom_model_5/dense_46/BiasAdd/ReadVariableOp?
custom_model_5/dense_46/BiasAddBiasAdd(custom_model_5/dense_46/MatMul:product:06custom_model_5/dense_46/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2!
custom_model_5/dense_46/BiasAdd?
custom_model_5/dense_46/ReluRelu(custom_model_5/dense_46/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
custom_model_5/dense_46/Relu?
-custom_model_5/dense_47/MatMul/ReadVariableOpReadVariableOp6custom_model_5_dense_47_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02/
-custom_model_5/dense_47/MatMul/ReadVariableOp?
custom_model_5/dense_47/MatMulMatMul*custom_model_5/dense_46/Relu:activations:05custom_model_5/dense_47/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2 
custom_model_5/dense_47/MatMul?
.custom_model_5/dense_47/BiasAdd/ReadVariableOpReadVariableOp7custom_model_5_dense_47_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype020
.custom_model_5/dense_47/BiasAdd/ReadVariableOp?
custom_model_5/dense_47/BiasAddBiasAdd(custom_model_5/dense_47/MatMul:product:06custom_model_5/dense_47/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2!
custom_model_5/dense_47/BiasAdd?
custom_model_5/dense_47/ReluRelu(custom_model_5/dense_47/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
custom_model_5/dense_47/Relu?
-custom_model_5/dense_49/MatMul/ReadVariableOpReadVariableOp6custom_model_5_dense_49_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02/
-custom_model_5/dense_49/MatMul/ReadVariableOp?
custom_model_5/dense_49/MatMulMatMul(custom_model_5/dense_48/BiasAdd:output:05custom_model_5/dense_49/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2 
custom_model_5/dense_49/MatMul?
.custom_model_5/dense_49/BiasAdd/ReadVariableOpReadVariableOp7custom_model_5_dense_49_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype020
.custom_model_5/dense_49/BiasAdd/ReadVariableOp?
custom_model_5/dense_49/BiasAddBiasAdd(custom_model_5/dense_49/MatMul:product:06custom_model_5/dense_49/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2!
custom_model_5/dense_49/BiasAdd?
'custom_model_5/tf.concat_17/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2)
'custom_model_5/tf.concat_17/concat/axis?
"custom_model_5/tf.concat_17/concatConcatV2*custom_model_5/dense_47/Relu:activations:0(custom_model_5/dense_49/BiasAdd:output:00custom_model_5/tf.concat_17/concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????2$
"custom_model_5/tf.concat_17/concat?
-custom_model_5/dense_50/MatMul/ReadVariableOpReadVariableOp6custom_model_5_dense_50_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02/
-custom_model_5/dense_50/MatMul/ReadVariableOp?
custom_model_5/dense_50/MatMulMatMul+custom_model_5/tf.concat_17/concat:output:05custom_model_5/dense_50/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2 
custom_model_5/dense_50/MatMul?
.custom_model_5/dense_50/BiasAdd/ReadVariableOpReadVariableOp7custom_model_5_dense_50_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype020
.custom_model_5/dense_50/BiasAdd/ReadVariableOp?
custom_model_5/dense_50/BiasAddBiasAdd(custom_model_5/dense_50/MatMul:product:06custom_model_5/dense_50/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2!
custom_model_5/dense_50/BiasAdd?
!custom_model_5/tf.nn.relu_15/ReluRelu(custom_model_5/dense_50/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2#
!custom_model_5/tf.nn.relu_15/Relu?
-custom_model_5/dense_51/MatMul/ReadVariableOpReadVariableOp6custom_model_5_dense_51_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02/
-custom_model_5/dense_51/MatMul/ReadVariableOp?
custom_model_5/dense_51/MatMulMatMul/custom_model_5/tf.nn.relu_15/Relu:activations:05custom_model_5/dense_51/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2 
custom_model_5/dense_51/MatMul?
.custom_model_5/dense_51/BiasAdd/ReadVariableOpReadVariableOp7custom_model_5_dense_51_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype020
.custom_model_5/dense_51/BiasAdd/ReadVariableOp?
custom_model_5/dense_51/BiasAddBiasAdd(custom_model_5/dense_51/MatMul:product:06custom_model_5/dense_51/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2!
custom_model_5/dense_51/BiasAdd?
,custom_model_5/tf.__operators__.add_34/AddV2AddV2(custom_model_5/dense_51/BiasAdd:output:0/custom_model_5/tf.nn.relu_15/Relu:activations:0*
T0*(
_output_shapes
:??????????2.
,custom_model_5/tf.__operators__.add_34/AddV2?
!custom_model_5/tf.nn.relu_16/ReluRelu0custom_model_5/tf.__operators__.add_34/AddV2:z:0*
T0*(
_output_shapes
:??????????2#
!custom_model_5/tf.nn.relu_16/Relu?
-custom_model_5/dense_52/MatMul/ReadVariableOpReadVariableOp6custom_model_5_dense_52_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02/
-custom_model_5/dense_52/MatMul/ReadVariableOp?
custom_model_5/dense_52/MatMulMatMul/custom_model_5/tf.nn.relu_16/Relu:activations:05custom_model_5/dense_52/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2 
custom_model_5/dense_52/MatMul?
.custom_model_5/dense_52/BiasAdd/ReadVariableOpReadVariableOp7custom_model_5_dense_52_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype020
.custom_model_5/dense_52/BiasAdd/ReadVariableOp?
custom_model_5/dense_52/BiasAddBiasAdd(custom_model_5/dense_52/MatMul:product:06custom_model_5/dense_52/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2!
custom_model_5/dense_52/BiasAdd?
,custom_model_5/tf.__operators__.add_35/AddV2AddV2(custom_model_5/dense_52/BiasAdd:output:0/custom_model_5/tf.nn.relu_16/Relu:activations:0*
T0*(
_output_shapes
:??????????2.
,custom_model_5/tf.__operators__.add_35/AddV2?
!custom_model_5/tf.nn.relu_17/ReluRelu0custom_model_5/tf.__operators__.add_35/AddV2:z:0*
T0*(
_output_shapes
:??????????2#
!custom_model_5/tf.nn.relu_17/Relu?
Acustom_model_5/normalize_5/normalization_5/Reshape/ReadVariableOpReadVariableOpJcustom_model_5_normalize_5_normalization_5_reshape_readvariableop_resource*
_output_shapes	
:?*
dtype02C
Acustom_model_5/normalize_5/normalization_5/Reshape/ReadVariableOp?
8custom_model_5/normalize_5/normalization_5/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2:
8custom_model_5/normalize_5/normalization_5/Reshape/shape?
2custom_model_5/normalize_5/normalization_5/ReshapeReshapeIcustom_model_5/normalize_5/normalization_5/Reshape/ReadVariableOp:value:0Acustom_model_5/normalize_5/normalization_5/Reshape/shape:output:0*
T0*
_output_shapes
:	?24
2custom_model_5/normalize_5/normalization_5/Reshape?
Ccustom_model_5/normalize_5/normalization_5/Reshape_1/ReadVariableOpReadVariableOpLcustom_model_5_normalize_5_normalization_5_reshape_1_readvariableop_resource*
_output_shapes	
:?*
dtype02E
Ccustom_model_5/normalize_5/normalization_5/Reshape_1/ReadVariableOp?
:custom_model_5/normalize_5/normalization_5/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2<
:custom_model_5/normalize_5/normalization_5/Reshape_1/shape?
4custom_model_5/normalize_5/normalization_5/Reshape_1ReshapeKcustom_model_5/normalize_5/normalization_5/Reshape_1/ReadVariableOp:value:0Ccustom_model_5/normalize_5/normalization_5/Reshape_1/shape:output:0*
T0*
_output_shapes
:	?26
4custom_model_5/normalize_5/normalization_5/Reshape_1?
.custom_model_5/normalize_5/normalization_5/subSub/custom_model_5/tf.nn.relu_17/Relu:activations:0;custom_model_5/normalize_5/normalization_5/Reshape:output:0*
T0*(
_output_shapes
:??????????20
.custom_model_5/normalize_5/normalization_5/sub?
/custom_model_5/normalize_5/normalization_5/SqrtSqrt=custom_model_5/normalize_5/normalization_5/Reshape_1:output:0*
T0*
_output_shapes
:	?21
/custom_model_5/normalize_5/normalization_5/Sqrt?
4custom_model_5/normalize_5/normalization_5/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???326
4custom_model_5/normalize_5/normalization_5/Maximum/y?
2custom_model_5/normalize_5/normalization_5/MaximumMaximum3custom_model_5/normalize_5/normalization_5/Sqrt:y:0=custom_model_5/normalize_5/normalization_5/Maximum/y:output:0*
T0*
_output_shapes
:	?24
2custom_model_5/normalize_5/normalization_5/Maximum?
2custom_model_5/normalize_5/normalization_5/truedivRealDiv2custom_model_5/normalize_5/normalization_5/sub:z:06custom_model_5/normalize_5/normalization_5/Maximum:z:0*
T0*(
_output_shapes
:??????????24
2custom_model_5/normalize_5/normalization_5/truediv?
-custom_model_5/dense_53/MatMul/ReadVariableOpReadVariableOp6custom_model_5_dense_53_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02/
-custom_model_5/dense_53/MatMul/ReadVariableOp?
custom_model_5/dense_53/MatMulMatMul6custom_model_5/normalize_5/normalization_5/truediv:z:05custom_model_5/dense_53/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2 
custom_model_5/dense_53/MatMul?
.custom_model_5/dense_53/BiasAdd/ReadVariableOpReadVariableOp7custom_model_5_dense_53_biasadd_readvariableop_resource*
_output_shapes
:*
dtype020
.custom_model_5/dense_53/BiasAdd/ReadVariableOp?
custom_model_5/dense_53/BiasAddBiasAdd(custom_model_5/dense_53/MatMul:product:06custom_model_5/dense_53/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2!
custom_model_5/dense_53/BiasAdd?
IdentityIdentity(custom_model_5/dense_53/BiasAdd:output:0/^custom_model_5/dense_45/BiasAdd/ReadVariableOp.^custom_model_5/dense_45/MatMul/ReadVariableOp/^custom_model_5/dense_46/BiasAdd/ReadVariableOp.^custom_model_5/dense_46/MatMul/ReadVariableOp/^custom_model_5/dense_47/BiasAdd/ReadVariableOp.^custom_model_5/dense_47/MatMul/ReadVariableOp/^custom_model_5/dense_48/BiasAdd/ReadVariableOp.^custom_model_5/dense_48/MatMul/ReadVariableOp/^custom_model_5/dense_49/BiasAdd/ReadVariableOp.^custom_model_5/dense_49/MatMul/ReadVariableOp/^custom_model_5/dense_50/BiasAdd/ReadVariableOp.^custom_model_5/dense_50/MatMul/ReadVariableOp/^custom_model_5/dense_51/BiasAdd/ReadVariableOp.^custom_model_5/dense_51/MatMul/ReadVariableOp/^custom_model_5/dense_52/BiasAdd/ReadVariableOp.^custom_model_5/dense_52/MatMul/ReadVariableOp/^custom_model_5/dense_53/BiasAdd/ReadVariableOp.^custom_model_5/dense_53/MatMul/ReadVariableOp6^custom_model_5/model_10/embedding_30/embedding_lookup6^custom_model_5/model_10/embedding_31/embedding_lookup6^custom_model_5/model_10/embedding_32/embedding_lookup6^custom_model_5/model_11/embedding_33/embedding_lookup6^custom_model_5/model_11/embedding_34/embedding_lookup6^custom_model_5/model_11/embedding_35/embedding_lookupB^custom_model_5/normalize_5/normalization_5/Reshape/ReadVariableOpD^custom_model_5/normalize_5/normalization_5/Reshape_1/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????:?????????:?????????
: : :::: :::: : ::::::::::::::::::::2`
.custom_model_5/dense_45/BiasAdd/ReadVariableOp.custom_model_5/dense_45/BiasAdd/ReadVariableOp2^
-custom_model_5/dense_45/MatMul/ReadVariableOp-custom_model_5/dense_45/MatMul/ReadVariableOp2`
.custom_model_5/dense_46/BiasAdd/ReadVariableOp.custom_model_5/dense_46/BiasAdd/ReadVariableOp2^
-custom_model_5/dense_46/MatMul/ReadVariableOp-custom_model_5/dense_46/MatMul/ReadVariableOp2`
.custom_model_5/dense_47/BiasAdd/ReadVariableOp.custom_model_5/dense_47/BiasAdd/ReadVariableOp2^
-custom_model_5/dense_47/MatMul/ReadVariableOp-custom_model_5/dense_47/MatMul/ReadVariableOp2`
.custom_model_5/dense_48/BiasAdd/ReadVariableOp.custom_model_5/dense_48/BiasAdd/ReadVariableOp2^
-custom_model_5/dense_48/MatMul/ReadVariableOp-custom_model_5/dense_48/MatMul/ReadVariableOp2`
.custom_model_5/dense_49/BiasAdd/ReadVariableOp.custom_model_5/dense_49/BiasAdd/ReadVariableOp2^
-custom_model_5/dense_49/MatMul/ReadVariableOp-custom_model_5/dense_49/MatMul/ReadVariableOp2`
.custom_model_5/dense_50/BiasAdd/ReadVariableOp.custom_model_5/dense_50/BiasAdd/ReadVariableOp2^
-custom_model_5/dense_50/MatMul/ReadVariableOp-custom_model_5/dense_50/MatMul/ReadVariableOp2`
.custom_model_5/dense_51/BiasAdd/ReadVariableOp.custom_model_5/dense_51/BiasAdd/ReadVariableOp2^
-custom_model_5/dense_51/MatMul/ReadVariableOp-custom_model_5/dense_51/MatMul/ReadVariableOp2`
.custom_model_5/dense_52/BiasAdd/ReadVariableOp.custom_model_5/dense_52/BiasAdd/ReadVariableOp2^
-custom_model_5/dense_52/MatMul/ReadVariableOp-custom_model_5/dense_52/MatMul/ReadVariableOp2`
.custom_model_5/dense_53/BiasAdd/ReadVariableOp.custom_model_5/dense_53/BiasAdd/ReadVariableOp2^
-custom_model_5/dense_53/MatMul/ReadVariableOp-custom_model_5/dense_53/MatMul/ReadVariableOp2n
5custom_model_5/model_10/embedding_30/embedding_lookup5custom_model_5/model_10/embedding_30/embedding_lookup2n
5custom_model_5/model_10/embedding_31/embedding_lookup5custom_model_5/model_10/embedding_31/embedding_lookup2n
5custom_model_5/model_10/embedding_32/embedding_lookup5custom_model_5/model_10/embedding_32/embedding_lookup2n
5custom_model_5/model_11/embedding_33/embedding_lookup5custom_model_5/model_11/embedding_33/embedding_lookup2n
5custom_model_5/model_11/embedding_34/embedding_lookup5custom_model_5/model_11/embedding_34/embedding_lookup2n
5custom_model_5/model_11/embedding_35/embedding_lookup5custom_model_5/model_11/embedding_35/embedding_lookup2?
Acustom_model_5/normalize_5/normalization_5/Reshape/ReadVariableOpAcustom_model_5/normalize_5/normalization_5/Reshape/ReadVariableOp2?
Ccustom_model_5/normalize_5/normalization_5/Reshape_1/ReadVariableOpCcustom_model_5/normalize_5/normalization_5/Reshape_1/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_namecards0:OK
'
_output_shapes
:?????????
 
_user_specified_namecards1:MI
'
_output_shapes
:?????????


_user_specified_namebets:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
u
/__inference_embedding_34_layer_call_fn_41027688

inputs
unknown
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *S
fNRL
J__inference_embedding_34_layer_call_and_return_conditional_losses_410256652
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????:22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
+__inference_dense_47_layer_call_fn_41027443

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_dense_47_layer_call_and_return_conditional_losses_410259852
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
u
/__inference_embedding_30_layer_call_fn_41027609

inputs
unknown
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *S
fNRL
J__inference_embedding_30_layer_call_and_return_conditional_losses_410254152
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????:22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
F__inference_dense_51_layer_call_and_return_conditional_losses_41026066

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
F__inference_dense_53_layer_call_and_return_conditional_losses_41026155

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
??
?
L__inference_custom_model_5_layer_call_and_return_conditional_losses_41027006

inputs_0_0

inputs_0_1
inputs_1+
'tf_math_greater_equal_17_greaterequal_y4
0model_10_tf_math_greater_equal_15_greaterequal_y3
/model_10_embedding_32_embedding_lookup_410268563
/model_10_embedding_30_embedding_lookup_410268623
/model_10_embedding_31_embedding_lookup_410268704
0model_11_tf_math_greater_equal_16_greaterequal_y3
/model_11_embedding_35_embedding_lookup_410268943
/model_11_embedding_33_embedding_lookup_410269003
/model_11_embedding_34_embedding_lookup_41026908/
+tf_clip_by_value_17_clip_by_value_minimum_y'
#tf_clip_by_value_17_clip_by_value_y+
'dense_45_matmul_readvariableop_resource,
(dense_45_biasadd_readvariableop_resource+
'dense_48_matmul_readvariableop_resource,
(dense_48_biasadd_readvariableop_resource+
'dense_46_matmul_readvariableop_resource,
(dense_46_biasadd_readvariableop_resource+
'dense_47_matmul_readvariableop_resource,
(dense_47_biasadd_readvariableop_resource+
'dense_49_matmul_readvariableop_resource,
(dense_49_biasadd_readvariableop_resource+
'dense_50_matmul_readvariableop_resource,
(dense_50_biasadd_readvariableop_resource+
'dense_51_matmul_readvariableop_resource,
(dense_51_biasadd_readvariableop_resource+
'dense_52_matmul_readvariableop_resource,
(dense_52_biasadd_readvariableop_resource?
;normalize_5_normalization_5_reshape_readvariableop_resourceA
=normalize_5_normalization_5_reshape_1_readvariableop_resource+
'dense_53_matmul_readvariableop_resource,
(dense_53_biasadd_readvariableop_resource
identity??dense_45/BiasAdd/ReadVariableOp?dense_45/MatMul/ReadVariableOp?dense_46/BiasAdd/ReadVariableOp?dense_46/MatMul/ReadVariableOp?dense_47/BiasAdd/ReadVariableOp?dense_47/MatMul/ReadVariableOp?dense_48/BiasAdd/ReadVariableOp?dense_48/MatMul/ReadVariableOp?dense_49/BiasAdd/ReadVariableOp?dense_49/MatMul/ReadVariableOp?dense_50/BiasAdd/ReadVariableOp?dense_50/MatMul/ReadVariableOp?dense_51/BiasAdd/ReadVariableOp?dense_51/MatMul/ReadVariableOp?dense_52/BiasAdd/ReadVariableOp?dense_52/MatMul/ReadVariableOp?dense_53/BiasAdd/ReadVariableOp?dense_53/MatMul/ReadVariableOp?&model_10/embedding_30/embedding_lookup?&model_10/embedding_31/embedding_lookup?&model_10/embedding_32/embedding_lookup?&model_11/embedding_33/embedding_lookup?&model_11/embedding_34/embedding_lookup?&model_11/embedding_35/embedding_lookup?2normalize_5/normalization_5/Reshape/ReadVariableOp?4normalize_5/normalization_5/Reshape_1/ReadVariableOp?
%tf.math.greater_equal_17/GreaterEqualGreaterEqualinputs_1'tf_math_greater_equal_17_greaterequal_y*
T0*'
_output_shapes
:?????????
2'
%tf.math.greater_equal_17/GreaterEqual?
model_10/flatten_10/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
model_10/flatten_10/Const?
model_10/flatten_10/ReshapeReshape
inputs_0_0"model_10/flatten_10/Const:output:0*
T0*'
_output_shapes
:?????????2
model_10/flatten_10/Reshape?
4model_10/tf.clip_by_value_15/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 * $tI26
4model_10/tf.clip_by_value_15/clip_by_value/Minimum/y?
2model_10/tf.clip_by_value_15/clip_by_value/MinimumMinimum$model_10/flatten_10/Reshape:output:0=model_10/tf.clip_by_value_15/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????24
2model_10/tf.clip_by_value_15/clip_by_value/Minimum?
,model_10/tf.clip_by_value_15/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2.
,model_10/tf.clip_by_value_15/clip_by_value/y?
*model_10/tf.clip_by_value_15/clip_by_valueMaximum6model_10/tf.clip_by_value_15/clip_by_value/Minimum:z:05model_10/tf.clip_by_value_15/clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????2,
*model_10/tf.clip_by_value_15/clip_by_value?
-model_10/tf.compat.v1.floor_div_10/FloorDiv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2/
-model_10/tf.compat.v1.floor_div_10/FloorDiv/y?
+model_10/tf.compat.v1.floor_div_10/FloorDivFloorDiv.model_10/tf.clip_by_value_15/clip_by_value:z:06model_10/tf.compat.v1.floor_div_10/FloorDiv/y:output:0*
T0*'
_output_shapes
:?????????2-
+model_10/tf.compat.v1.floor_div_10/FloorDiv?
.model_10/tf.math.greater_equal_15/GreaterEqualGreaterEqual$model_10/flatten_10/Reshape:output:00model_10_tf_math_greater_equal_15_greaterequal_y*
T0*'
_output_shapes
:?????????20
.model_10/tf.math.greater_equal_15/GreaterEqual?
'model_10/tf.math.floormod_10/FloorMod/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2)
'model_10/tf.math.floormod_10/FloorMod/y?
%model_10/tf.math.floormod_10/FloorModFloorMod.model_10/tf.clip_by_value_15/clip_by_value:z:00model_10/tf.math.floormod_10/FloorMod/y:output:0*
T0*'
_output_shapes
:?????????2'
%model_10/tf.math.floormod_10/FloorMod?
model_10/embedding_32/CastCast.model_10/tf.clip_by_value_15/clip_by_value:z:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2
model_10/embedding_32/Cast?
&model_10/embedding_32/embedding_lookupResourceGather/model_10_embedding_32_embedding_lookup_41026856model_10/embedding_32/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*B
_class8
64loc:@model_10/embedding_32/embedding_lookup/41026856*,
_output_shapes
:??????????*
dtype02(
&model_10/embedding_32/embedding_lookup?
/model_10/embedding_32/embedding_lookup/IdentityIdentity/model_10/embedding_32/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*B
_class8
64loc:@model_10/embedding_32/embedding_lookup/41026856*,
_output_shapes
:??????????21
/model_10/embedding_32/embedding_lookup/Identity?
1model_10/embedding_32/embedding_lookup/Identity_1Identity8model_10/embedding_32/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????23
1model_10/embedding_32/embedding_lookup/Identity_1?
model_10/embedding_30/CastCast/model_10/tf.compat.v1.floor_div_10/FloorDiv:z:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2
model_10/embedding_30/Cast?
&model_10/embedding_30/embedding_lookupResourceGather/model_10_embedding_30_embedding_lookup_41026862model_10/embedding_30/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*B
_class8
64loc:@model_10/embedding_30/embedding_lookup/41026862*,
_output_shapes
:??????????*
dtype02(
&model_10/embedding_30/embedding_lookup?
/model_10/embedding_30/embedding_lookup/IdentityIdentity/model_10/embedding_30/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*B
_class8
64loc:@model_10/embedding_30/embedding_lookup/41026862*,
_output_shapes
:??????????21
/model_10/embedding_30/embedding_lookup/Identity?
1model_10/embedding_30/embedding_lookup/Identity_1Identity8model_10/embedding_30/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????23
1model_10/embedding_30/embedding_lookup/Identity_1?
model_10/tf.cast_15/CastCast2model_10/tf.math.greater_equal_15/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2
model_10/tf.cast_15/Cast?
&model_10/tf.__operators__.add_30/AddV2AddV2:model_10/embedding_32/embedding_lookup/Identity_1:output:0:model_10/embedding_30/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:??????????2(
&model_10/tf.__operators__.add_30/AddV2?
model_10/embedding_31/CastCast)model_10/tf.math.floormod_10/FloorMod:z:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2
model_10/embedding_31/Cast?
&model_10/embedding_31/embedding_lookupResourceGather/model_10_embedding_31_embedding_lookup_41026870model_10/embedding_31/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*B
_class8
64loc:@model_10/embedding_31/embedding_lookup/41026870*,
_output_shapes
:??????????*
dtype02(
&model_10/embedding_31/embedding_lookup?
/model_10/embedding_31/embedding_lookup/IdentityIdentity/model_10/embedding_31/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*B
_class8
64loc:@model_10/embedding_31/embedding_lookup/41026870*,
_output_shapes
:??????????21
/model_10/embedding_31/embedding_lookup/Identity?
1model_10/embedding_31/embedding_lookup/Identity_1Identity8model_10/embedding_31/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????23
1model_10/embedding_31/embedding_lookup/Identity_1?
&model_10/tf.__operators__.add_31/AddV2AddV2*model_10/tf.__operators__.add_30/AddV2:z:0:model_10/embedding_31/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:??????????2(
&model_10/tf.__operators__.add_31/AddV2?
)model_10/tf.expand_dims_10/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2+
)model_10/tf.expand_dims_10/ExpandDims/dim?
%model_10/tf.expand_dims_10/ExpandDims
ExpandDimsmodel_10/tf.cast_15/Cast:y:02model_10/tf.expand_dims_10/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????2'
%model_10/tf.expand_dims_10/ExpandDims?
 model_10/tf.math.multiply_10/MulMul*model_10/tf.__operators__.add_31/AddV2:z:0.model_10/tf.expand_dims_10/ExpandDims:output:0*
T0*,
_output_shapes
:??????????2"
 model_10/tf.math.multiply_10/Mul?
4model_10/tf.math.reduce_sum_10/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :26
4model_10/tf.math.reduce_sum_10/Sum/reduction_indices?
"model_10/tf.math.reduce_sum_10/SumSum$model_10/tf.math.multiply_10/Mul:z:0=model_10/tf.math.reduce_sum_10/Sum/reduction_indices:output:0*
T0*(
_output_shapes
:??????????2$
"model_10/tf.math.reduce_sum_10/Sum?
model_11/flatten_11/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
model_11/flatten_11/Const?
model_11/flatten_11/ReshapeReshape
inputs_0_1"model_11/flatten_11/Const:output:0*
T0*'
_output_shapes
:?????????2
model_11/flatten_11/Reshape?
4model_11/tf.clip_by_value_16/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 * $tI26
4model_11/tf.clip_by_value_16/clip_by_value/Minimum/y?
2model_11/tf.clip_by_value_16/clip_by_value/MinimumMinimum$model_11/flatten_11/Reshape:output:0=model_11/tf.clip_by_value_16/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????24
2model_11/tf.clip_by_value_16/clip_by_value/Minimum?
,model_11/tf.clip_by_value_16/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2.
,model_11/tf.clip_by_value_16/clip_by_value/y?
*model_11/tf.clip_by_value_16/clip_by_valueMaximum6model_11/tf.clip_by_value_16/clip_by_value/Minimum:z:05model_11/tf.clip_by_value_16/clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????2,
*model_11/tf.clip_by_value_16/clip_by_value?
-model_11/tf.compat.v1.floor_div_11/FloorDiv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2/
-model_11/tf.compat.v1.floor_div_11/FloorDiv/y?
+model_11/tf.compat.v1.floor_div_11/FloorDivFloorDiv.model_11/tf.clip_by_value_16/clip_by_value:z:06model_11/tf.compat.v1.floor_div_11/FloorDiv/y:output:0*
T0*'
_output_shapes
:?????????2-
+model_11/tf.compat.v1.floor_div_11/FloorDiv?
.model_11/tf.math.greater_equal_16/GreaterEqualGreaterEqual$model_11/flatten_11/Reshape:output:00model_11_tf_math_greater_equal_16_greaterequal_y*
T0*'
_output_shapes
:?????????20
.model_11/tf.math.greater_equal_16/GreaterEqual?
'model_11/tf.math.floormod_11/FloorMod/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2)
'model_11/tf.math.floormod_11/FloorMod/y?
%model_11/tf.math.floormod_11/FloorModFloorMod.model_11/tf.clip_by_value_16/clip_by_value:z:00model_11/tf.math.floormod_11/FloorMod/y:output:0*
T0*'
_output_shapes
:?????????2'
%model_11/tf.math.floormod_11/FloorMod?
model_11/embedding_35/CastCast.model_11/tf.clip_by_value_16/clip_by_value:z:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2
model_11/embedding_35/Cast?
&model_11/embedding_35/embedding_lookupResourceGather/model_11_embedding_35_embedding_lookup_41026894model_11/embedding_35/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*B
_class8
64loc:@model_11/embedding_35/embedding_lookup/41026894*,
_output_shapes
:??????????*
dtype02(
&model_11/embedding_35/embedding_lookup?
/model_11/embedding_35/embedding_lookup/IdentityIdentity/model_11/embedding_35/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*B
_class8
64loc:@model_11/embedding_35/embedding_lookup/41026894*,
_output_shapes
:??????????21
/model_11/embedding_35/embedding_lookup/Identity?
1model_11/embedding_35/embedding_lookup/Identity_1Identity8model_11/embedding_35/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????23
1model_11/embedding_35/embedding_lookup/Identity_1?
model_11/embedding_33/CastCast/model_11/tf.compat.v1.floor_div_11/FloorDiv:z:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2
model_11/embedding_33/Cast?
&model_11/embedding_33/embedding_lookupResourceGather/model_11_embedding_33_embedding_lookup_41026900model_11/embedding_33/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*B
_class8
64loc:@model_11/embedding_33/embedding_lookup/41026900*,
_output_shapes
:??????????*
dtype02(
&model_11/embedding_33/embedding_lookup?
/model_11/embedding_33/embedding_lookup/IdentityIdentity/model_11/embedding_33/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*B
_class8
64loc:@model_11/embedding_33/embedding_lookup/41026900*,
_output_shapes
:??????????21
/model_11/embedding_33/embedding_lookup/Identity?
1model_11/embedding_33/embedding_lookup/Identity_1Identity8model_11/embedding_33/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????23
1model_11/embedding_33/embedding_lookup/Identity_1?
model_11/tf.cast_16/CastCast2model_11/tf.math.greater_equal_16/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2
model_11/tf.cast_16/Cast?
&model_11/tf.__operators__.add_32/AddV2AddV2:model_11/embedding_35/embedding_lookup/Identity_1:output:0:model_11/embedding_33/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:??????????2(
&model_11/tf.__operators__.add_32/AddV2?
model_11/embedding_34/CastCast)model_11/tf.math.floormod_11/FloorMod:z:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2
model_11/embedding_34/Cast?
&model_11/embedding_34/embedding_lookupResourceGather/model_11_embedding_34_embedding_lookup_41026908model_11/embedding_34/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*B
_class8
64loc:@model_11/embedding_34/embedding_lookup/41026908*,
_output_shapes
:??????????*
dtype02(
&model_11/embedding_34/embedding_lookup?
/model_11/embedding_34/embedding_lookup/IdentityIdentity/model_11/embedding_34/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*B
_class8
64loc:@model_11/embedding_34/embedding_lookup/41026908*,
_output_shapes
:??????????21
/model_11/embedding_34/embedding_lookup/Identity?
1model_11/embedding_34/embedding_lookup/Identity_1Identity8model_11/embedding_34/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????23
1model_11/embedding_34/embedding_lookup/Identity_1?
&model_11/tf.__operators__.add_33/AddV2AddV2*model_11/tf.__operators__.add_32/AddV2:z:0:model_11/embedding_34/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:??????????2(
&model_11/tf.__operators__.add_33/AddV2?
)model_11/tf.expand_dims_11/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2+
)model_11/tf.expand_dims_11/ExpandDims/dim?
%model_11/tf.expand_dims_11/ExpandDims
ExpandDimsmodel_11/tf.cast_16/Cast:y:02model_11/tf.expand_dims_11/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????2'
%model_11/tf.expand_dims_11/ExpandDims?
 model_11/tf.math.multiply_11/MulMul*model_11/tf.__operators__.add_33/AddV2:z:0.model_11/tf.expand_dims_11/ExpandDims:output:0*
T0*,
_output_shapes
:??????????2"
 model_11/tf.math.multiply_11/Mul?
4model_11/tf.math.reduce_sum_11/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :26
4model_11/tf.math.reduce_sum_11/Sum/reduction_indices?
"model_11/tf.math.reduce_sum_11/SumSum$model_11/tf.math.multiply_11/Mul:z:0=model_11/tf.math.reduce_sum_11/Sum/reduction_indices:output:0*
T0*(
_output_shapes
:??????????2$
"model_11/tf.math.reduce_sum_11/Sum?
)tf.clip_by_value_17/clip_by_value/MinimumMinimuminputs_1+tf_clip_by_value_17_clip_by_value_minimum_y*
T0*'
_output_shapes
:?????????
2+
)tf.clip_by_value_17/clip_by_value/Minimum?
!tf.clip_by_value_17/clip_by_valueMaximum-tf.clip_by_value_17/clip_by_value/Minimum:z:0#tf_clip_by_value_17_clip_by_value_y*
T0*'
_output_shapes
:?????????
2#
!tf.clip_by_value_17/clip_by_value?
tf.cast_17/CastCast)tf.math.greater_equal_17/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????
2
tf.cast_17/Castv
tf.concat_15/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
tf.concat_15/concat/axis?
tf.concat_15/concatConcatV2+model_10/tf.math.reduce_sum_10/Sum:output:0+model_11/tf.math.reduce_sum_11/Sum:output:0!tf.concat_15/concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????2
tf.concat_15/concat
tf.concat_16/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
tf.concat_16/concat/axis?
tf.concat_16/concatConcatV2%tf.clip_by_value_17/clip_by_value:z:0tf.cast_17/Cast:y:0!tf.concat_16/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????2
tf.concat_16/concat?
dense_45/MatMul/ReadVariableOpReadVariableOp'dense_45_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02 
dense_45/MatMul/ReadVariableOp?
dense_45/MatMulMatMultf.concat_15/concat:output:0&dense_45/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_45/MatMul?
dense_45/BiasAdd/ReadVariableOpReadVariableOp(dense_45_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
dense_45/BiasAdd/ReadVariableOp?
dense_45/BiasAddBiasAdddense_45/MatMul:product:0'dense_45/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_45/BiasAddt
dense_45/ReluReludense_45/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_45/Relu?
dense_48/MatMul/ReadVariableOpReadVariableOp'dense_48_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02 
dense_48/MatMul/ReadVariableOp?
dense_48/MatMulMatMultf.concat_16/concat:output:0&dense_48/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_48/MatMul?
dense_48/BiasAdd/ReadVariableOpReadVariableOp(dense_48_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
dense_48/BiasAdd/ReadVariableOp?
dense_48/BiasAddBiasAdddense_48/MatMul:product:0'dense_48/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_48/BiasAdd?
dense_46/MatMul/ReadVariableOpReadVariableOp'dense_46_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02 
dense_46/MatMul/ReadVariableOp?
dense_46/MatMulMatMuldense_45/Relu:activations:0&dense_46/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_46/MatMul?
dense_46/BiasAdd/ReadVariableOpReadVariableOp(dense_46_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
dense_46/BiasAdd/ReadVariableOp?
dense_46/BiasAddBiasAdddense_46/MatMul:product:0'dense_46/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_46/BiasAddt
dense_46/ReluReludense_46/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_46/Relu?
dense_47/MatMul/ReadVariableOpReadVariableOp'dense_47_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02 
dense_47/MatMul/ReadVariableOp?
dense_47/MatMulMatMuldense_46/Relu:activations:0&dense_47/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_47/MatMul?
dense_47/BiasAdd/ReadVariableOpReadVariableOp(dense_47_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
dense_47/BiasAdd/ReadVariableOp?
dense_47/BiasAddBiasAdddense_47/MatMul:product:0'dense_47/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_47/BiasAddt
dense_47/ReluReludense_47/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_47/Relu?
dense_49/MatMul/ReadVariableOpReadVariableOp'dense_49_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02 
dense_49/MatMul/ReadVariableOp?
dense_49/MatMulMatMuldense_48/BiasAdd:output:0&dense_49/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_49/MatMul?
dense_49/BiasAdd/ReadVariableOpReadVariableOp(dense_49_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
dense_49/BiasAdd/ReadVariableOp?
dense_49/BiasAddBiasAdddense_49/MatMul:product:0'dense_49/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_49/BiasAdd
tf.concat_17/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
tf.concat_17/concat/axis?
tf.concat_17/concatConcatV2dense_47/Relu:activations:0dense_49/BiasAdd:output:0!tf.concat_17/concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????2
tf.concat_17/concat?
dense_50/MatMul/ReadVariableOpReadVariableOp'dense_50_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02 
dense_50/MatMul/ReadVariableOp?
dense_50/MatMulMatMultf.concat_17/concat:output:0&dense_50/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_50/MatMul?
dense_50/BiasAdd/ReadVariableOpReadVariableOp(dense_50_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
dense_50/BiasAdd/ReadVariableOp?
dense_50/BiasAddBiasAdddense_50/MatMul:product:0'dense_50/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_50/BiasAdd~
tf.nn.relu_15/ReluReludense_50/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
tf.nn.relu_15/Relu?
dense_51/MatMul/ReadVariableOpReadVariableOp'dense_51_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02 
dense_51/MatMul/ReadVariableOp?
dense_51/MatMulMatMul tf.nn.relu_15/Relu:activations:0&dense_51/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_51/MatMul?
dense_51/BiasAdd/ReadVariableOpReadVariableOp(dense_51_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
dense_51/BiasAdd/ReadVariableOp?
dense_51/BiasAddBiasAdddense_51/MatMul:product:0'dense_51/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_51/BiasAdd?
tf.__operators__.add_34/AddV2AddV2dense_51/BiasAdd:output:0 tf.nn.relu_15/Relu:activations:0*
T0*(
_output_shapes
:??????????2
tf.__operators__.add_34/AddV2?
tf.nn.relu_16/ReluRelu!tf.__operators__.add_34/AddV2:z:0*
T0*(
_output_shapes
:??????????2
tf.nn.relu_16/Relu?
dense_52/MatMul/ReadVariableOpReadVariableOp'dense_52_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02 
dense_52/MatMul/ReadVariableOp?
dense_52/MatMulMatMul tf.nn.relu_16/Relu:activations:0&dense_52/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_52/MatMul?
dense_52/BiasAdd/ReadVariableOpReadVariableOp(dense_52_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
dense_52/BiasAdd/ReadVariableOp?
dense_52/BiasAddBiasAdddense_52/MatMul:product:0'dense_52/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_52/BiasAdd?
tf.__operators__.add_35/AddV2AddV2dense_52/BiasAdd:output:0 tf.nn.relu_16/Relu:activations:0*
T0*(
_output_shapes
:??????????2
tf.__operators__.add_35/AddV2?
tf.nn.relu_17/ReluRelu!tf.__operators__.add_35/AddV2:z:0*
T0*(
_output_shapes
:??????????2
tf.nn.relu_17/Relu?
2normalize_5/normalization_5/Reshape/ReadVariableOpReadVariableOp;normalize_5_normalization_5_reshape_readvariableop_resource*
_output_shapes	
:?*
dtype024
2normalize_5/normalization_5/Reshape/ReadVariableOp?
)normalize_5/normalization_5/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2+
)normalize_5/normalization_5/Reshape/shape?
#normalize_5/normalization_5/ReshapeReshape:normalize_5/normalization_5/Reshape/ReadVariableOp:value:02normalize_5/normalization_5/Reshape/shape:output:0*
T0*
_output_shapes
:	?2%
#normalize_5/normalization_5/Reshape?
4normalize_5/normalization_5/Reshape_1/ReadVariableOpReadVariableOp=normalize_5_normalization_5_reshape_1_readvariableop_resource*
_output_shapes	
:?*
dtype026
4normalize_5/normalization_5/Reshape_1/ReadVariableOp?
+normalize_5/normalization_5/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2-
+normalize_5/normalization_5/Reshape_1/shape?
%normalize_5/normalization_5/Reshape_1Reshape<normalize_5/normalization_5/Reshape_1/ReadVariableOp:value:04normalize_5/normalization_5/Reshape_1/shape:output:0*
T0*
_output_shapes
:	?2'
%normalize_5/normalization_5/Reshape_1?
normalize_5/normalization_5/subSub tf.nn.relu_17/Relu:activations:0,normalize_5/normalization_5/Reshape:output:0*
T0*(
_output_shapes
:??????????2!
normalize_5/normalization_5/sub?
 normalize_5/normalization_5/SqrtSqrt.normalize_5/normalization_5/Reshape_1:output:0*
T0*
_output_shapes
:	?2"
 normalize_5/normalization_5/Sqrt?
%normalize_5/normalization_5/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32'
%normalize_5/normalization_5/Maximum/y?
#normalize_5/normalization_5/MaximumMaximum$normalize_5/normalization_5/Sqrt:y:0.normalize_5/normalization_5/Maximum/y:output:0*
T0*
_output_shapes
:	?2%
#normalize_5/normalization_5/Maximum?
#normalize_5/normalization_5/truedivRealDiv#normalize_5/normalization_5/sub:z:0'normalize_5/normalization_5/Maximum:z:0*
T0*(
_output_shapes
:??????????2%
#normalize_5/normalization_5/truediv?
dense_53/MatMul/ReadVariableOpReadVariableOp'dense_53_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02 
dense_53/MatMul/ReadVariableOp?
dense_53/MatMulMatMul'normalize_5/normalization_5/truediv:z:0&dense_53/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_53/MatMul?
dense_53/BiasAdd/ReadVariableOpReadVariableOp(dense_53_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_53/BiasAdd/ReadVariableOp?
dense_53/BiasAddBiasAdddense_53/MatMul:product:0'dense_53/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_53/BiasAdd?
IdentityIdentitydense_53/BiasAdd:output:0 ^dense_45/BiasAdd/ReadVariableOp^dense_45/MatMul/ReadVariableOp ^dense_46/BiasAdd/ReadVariableOp^dense_46/MatMul/ReadVariableOp ^dense_47/BiasAdd/ReadVariableOp^dense_47/MatMul/ReadVariableOp ^dense_48/BiasAdd/ReadVariableOp^dense_48/MatMul/ReadVariableOp ^dense_49/BiasAdd/ReadVariableOp^dense_49/MatMul/ReadVariableOp ^dense_50/BiasAdd/ReadVariableOp^dense_50/MatMul/ReadVariableOp ^dense_51/BiasAdd/ReadVariableOp^dense_51/MatMul/ReadVariableOp ^dense_52/BiasAdd/ReadVariableOp^dense_52/MatMul/ReadVariableOp ^dense_53/BiasAdd/ReadVariableOp^dense_53/MatMul/ReadVariableOp'^model_10/embedding_30/embedding_lookup'^model_10/embedding_31/embedding_lookup'^model_10/embedding_32/embedding_lookup'^model_11/embedding_33/embedding_lookup'^model_11/embedding_34/embedding_lookup'^model_11/embedding_35/embedding_lookup3^normalize_5/normalization_5/Reshape/ReadVariableOp5^normalize_5/normalization_5/Reshape_1/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????:?????????:?????????
: : :::: :::: : ::::::::::::::::::::2B
dense_45/BiasAdd/ReadVariableOpdense_45/BiasAdd/ReadVariableOp2@
dense_45/MatMul/ReadVariableOpdense_45/MatMul/ReadVariableOp2B
dense_46/BiasAdd/ReadVariableOpdense_46/BiasAdd/ReadVariableOp2@
dense_46/MatMul/ReadVariableOpdense_46/MatMul/ReadVariableOp2B
dense_47/BiasAdd/ReadVariableOpdense_47/BiasAdd/ReadVariableOp2@
dense_47/MatMul/ReadVariableOpdense_47/MatMul/ReadVariableOp2B
dense_48/BiasAdd/ReadVariableOpdense_48/BiasAdd/ReadVariableOp2@
dense_48/MatMul/ReadVariableOpdense_48/MatMul/ReadVariableOp2B
dense_49/BiasAdd/ReadVariableOpdense_49/BiasAdd/ReadVariableOp2@
dense_49/MatMul/ReadVariableOpdense_49/MatMul/ReadVariableOp2B
dense_50/BiasAdd/ReadVariableOpdense_50/BiasAdd/ReadVariableOp2@
dense_50/MatMul/ReadVariableOpdense_50/MatMul/ReadVariableOp2B
dense_51/BiasAdd/ReadVariableOpdense_51/BiasAdd/ReadVariableOp2@
dense_51/MatMul/ReadVariableOpdense_51/MatMul/ReadVariableOp2B
dense_52/BiasAdd/ReadVariableOpdense_52/BiasAdd/ReadVariableOp2@
dense_52/MatMul/ReadVariableOpdense_52/MatMul/ReadVariableOp2B
dense_53/BiasAdd/ReadVariableOpdense_53/BiasAdd/ReadVariableOp2@
dense_53/MatMul/ReadVariableOpdense_53/MatMul/ReadVariableOp2P
&model_10/embedding_30/embedding_lookup&model_10/embedding_30/embedding_lookup2P
&model_10/embedding_31/embedding_lookup&model_10/embedding_31/embedding_lookup2P
&model_10/embedding_32/embedding_lookup&model_10/embedding_32/embedding_lookup2P
&model_11/embedding_33/embedding_lookup&model_11/embedding_33/embedding_lookup2P
&model_11/embedding_34/embedding_lookup&model_11/embedding_34/embedding_lookup2P
&model_11/embedding_35/embedding_lookup&model_11/embedding_35/embedding_lookup2h
2normalize_5/normalization_5/Reshape/ReadVariableOp2normalize_5/normalization_5/Reshape/ReadVariableOp2l
4normalize_5/normalization_5/Reshape_1/ReadVariableOp4normalize_5/normalization_5/Reshape_1/ReadVariableOp:S O
'
_output_shapes
:?????????
$
_user_specified_name
inputs/0/0:SO
'
_output_shapes
:?????????
$
_user_specified_name
inputs/0/1:QM
'
_output_shapes
:?????????

"
_user_specified_name
inputs/1:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?.
?
F__inference_model_10_layer_call_and_return_conditional_losses_41025570

inputs+
'tf_math_greater_equal_15_greaterequal_y
embedding_32_41025552
embedding_30_41025555
embedding_31_41025560
identity??$embedding_30/StatefulPartitionedCall?$embedding_31/StatefulPartitionedCall?$embedding_32/StatefulPartitionedCall?
flatten_10/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *Q
fLRJ
H__inference_flatten_10_layer_call_and_return_conditional_losses_410253652
flatten_10/PartitionedCall?
+tf.clip_by_value_15/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 * $tI2-
+tf.clip_by_value_15/clip_by_value/Minimum/y?
)tf.clip_by_value_15/clip_by_value/MinimumMinimum#flatten_10/PartitionedCall:output:04tf.clip_by_value_15/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2+
)tf.clip_by_value_15/clip_by_value/Minimum?
#tf.clip_by_value_15/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2%
#tf.clip_by_value_15/clip_by_value/y?
!tf.clip_by_value_15/clip_by_valueMaximum-tf.clip_by_value_15/clip_by_value/Minimum:z:0,tf.clip_by_value_15/clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????2#
!tf.clip_by_value_15/clip_by_value?
$tf.compat.v1.floor_div_10/FloorDiv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2&
$tf.compat.v1.floor_div_10/FloorDiv/y?
"tf.compat.v1.floor_div_10/FloorDivFloorDiv%tf.clip_by_value_15/clip_by_value:z:0-tf.compat.v1.floor_div_10/FloorDiv/y:output:0*
T0*'
_output_shapes
:?????????2$
"tf.compat.v1.floor_div_10/FloorDiv?
%tf.math.greater_equal_15/GreaterEqualGreaterEqual#flatten_10/PartitionedCall:output:0'tf_math_greater_equal_15_greaterequal_y*
T0*'
_output_shapes
:?????????2'
%tf.math.greater_equal_15/GreaterEqual?
tf.math.floormod_10/FloorMod/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2 
tf.math.floormod_10/FloorMod/y?
tf.math.floormod_10/FloorModFloorMod%tf.clip_by_value_15/clip_by_value:z:0'tf.math.floormod_10/FloorMod/y:output:0*
T0*'
_output_shapes
:?????????2
tf.math.floormod_10/FloorMod?
$embedding_32/StatefulPartitionedCallStatefulPartitionedCall%tf.clip_by_value_15/clip_by_value:z:0embedding_32_41025552*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *S
fNRL
J__inference_embedding_32_layer_call_and_return_conditional_losses_410253932&
$embedding_32/StatefulPartitionedCall?
$embedding_30/StatefulPartitionedCallStatefulPartitionedCall&tf.compat.v1.floor_div_10/FloorDiv:z:0embedding_30_41025555*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *S
fNRL
J__inference_embedding_30_layer_call_and_return_conditional_losses_410254152&
$embedding_30/StatefulPartitionedCall?
tf.cast_15/CastCast)tf.math.greater_equal_15/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2
tf.cast_15/Cast?
tf.__operators__.add_30/AddV2AddV2-embedding_32/StatefulPartitionedCall:output:0-embedding_30/StatefulPartitionedCall:output:0*
T0*,
_output_shapes
:??????????2
tf.__operators__.add_30/AddV2?
$embedding_31/StatefulPartitionedCallStatefulPartitionedCall tf.math.floormod_10/FloorMod:z:0embedding_31_41025560*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *S
fNRL
J__inference_embedding_31_layer_call_and_return_conditional_losses_410254392&
$embedding_31/StatefulPartitionedCall?
tf.__operators__.add_31/AddV2AddV2!tf.__operators__.add_30/AddV2:z:0-embedding_31/StatefulPartitionedCall:output:0*
T0*,
_output_shapes
:??????????2
tf.__operators__.add_31/AddV2?
 tf.expand_dims_10/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2"
 tf.expand_dims_10/ExpandDims/dim?
tf.expand_dims_10/ExpandDims
ExpandDimstf.cast_15/Cast:y:0)tf.expand_dims_10/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????2
tf.expand_dims_10/ExpandDims?
tf.math.multiply_10/MulMul!tf.__operators__.add_31/AddV2:z:0%tf.expand_dims_10/ExpandDims:output:0*
T0*,
_output_shapes
:??????????2
tf.math.multiply_10/Mul?
+tf.math.reduce_sum_10/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2-
+tf.math.reduce_sum_10/Sum/reduction_indices?
tf.math.reduce_sum_10/SumSumtf.math.multiply_10/Mul:z:04tf.math.reduce_sum_10/Sum/reduction_indices:output:0*
T0*(
_output_shapes
:??????????2
tf.math.reduce_sum_10/Sum?
IdentityIdentity"tf.math.reduce_sum_10/Sum:output:0%^embedding_30/StatefulPartitionedCall%^embedding_31/StatefulPartitionedCall%^embedding_32/StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:?????????: :::2L
$embedding_30/StatefulPartitionedCall$embedding_30/StatefulPartitionedCall2L
$embedding_31/StatefulPartitionedCall$embedding_31/StatefulPartitionedCall2L
$embedding_32/StatefulPartitionedCall$embedding_32/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: 
?
u
/__inference_embedding_31_layer_call_fn_41027626

inputs
unknown
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *S
fNRL
J__inference_embedding_31_layer_call_and_return_conditional_losses_410254392
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????:22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
F__inference_dense_48_layer_call_and_return_conditional_losses_41025931

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
J__inference_embedding_34_layer_call_and_return_conditional_losses_41027681

inputs
embedding_lookup_41027675
identity??embedding_lookup]
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:?????????2
Cast?
embedding_lookupResourceGatherembedding_lookup_41027675Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*,
_class"
 loc:@embedding_lookup/41027675*,
_output_shapes
:??????????*
dtype02
embedding_lookup?
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*,
_class"
 loc:@embedding_lookup/41027675*,
_output_shapes
:??????????2
embedding_lookup/Identity?
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????2
embedding_lookup/Identity_1?
IdentityIdentity$embedding_lookup/Identity_1:output:0^embedding_lookup*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????:2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?9
?
F__inference_model_10_layer_call_and_return_conditional_losses_41027228

inputs+
'tf_math_greater_equal_15_greaterequal_y*
&embedding_32_embedding_lookup_41027202*
&embedding_30_embedding_lookup_41027208*
&embedding_31_embedding_lookup_41027216
identity??embedding_30/embedding_lookup?embedding_31/embedding_lookup?embedding_32/embedding_lookupu
flatten_10/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
flatten_10/Const?
flatten_10/ReshapeReshapeinputsflatten_10/Const:output:0*
T0*'
_output_shapes
:?????????2
flatten_10/Reshape?
+tf.clip_by_value_15/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 * $tI2-
+tf.clip_by_value_15/clip_by_value/Minimum/y?
)tf.clip_by_value_15/clip_by_value/MinimumMinimumflatten_10/Reshape:output:04tf.clip_by_value_15/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2+
)tf.clip_by_value_15/clip_by_value/Minimum?
#tf.clip_by_value_15/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2%
#tf.clip_by_value_15/clip_by_value/y?
!tf.clip_by_value_15/clip_by_valueMaximum-tf.clip_by_value_15/clip_by_value/Minimum:z:0,tf.clip_by_value_15/clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????2#
!tf.clip_by_value_15/clip_by_value?
$tf.compat.v1.floor_div_10/FloorDiv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2&
$tf.compat.v1.floor_div_10/FloorDiv/y?
"tf.compat.v1.floor_div_10/FloorDivFloorDiv%tf.clip_by_value_15/clip_by_value:z:0-tf.compat.v1.floor_div_10/FloorDiv/y:output:0*
T0*'
_output_shapes
:?????????2$
"tf.compat.v1.floor_div_10/FloorDiv?
%tf.math.greater_equal_15/GreaterEqualGreaterEqualflatten_10/Reshape:output:0'tf_math_greater_equal_15_greaterequal_y*
T0*'
_output_shapes
:?????????2'
%tf.math.greater_equal_15/GreaterEqual?
tf.math.floormod_10/FloorMod/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2 
tf.math.floormod_10/FloorMod/y?
tf.math.floormod_10/FloorModFloorMod%tf.clip_by_value_15/clip_by_value:z:0'tf.math.floormod_10/FloorMod/y:output:0*
T0*'
_output_shapes
:?????????2
tf.math.floormod_10/FloorMod?
embedding_32/CastCast%tf.clip_by_value_15/clip_by_value:z:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2
embedding_32/Cast?
embedding_32/embedding_lookupResourceGather&embedding_32_embedding_lookup_41027202embedding_32/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*9
_class/
-+loc:@embedding_32/embedding_lookup/41027202*,
_output_shapes
:??????????*
dtype02
embedding_32/embedding_lookup?
&embedding_32/embedding_lookup/IdentityIdentity&embedding_32/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*9
_class/
-+loc:@embedding_32/embedding_lookup/41027202*,
_output_shapes
:??????????2(
&embedding_32/embedding_lookup/Identity?
(embedding_32/embedding_lookup/Identity_1Identity/embedding_32/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????2*
(embedding_32/embedding_lookup/Identity_1?
embedding_30/CastCast&tf.compat.v1.floor_div_10/FloorDiv:z:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2
embedding_30/Cast?
embedding_30/embedding_lookupResourceGather&embedding_30_embedding_lookup_41027208embedding_30/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*9
_class/
-+loc:@embedding_30/embedding_lookup/41027208*,
_output_shapes
:??????????*
dtype02
embedding_30/embedding_lookup?
&embedding_30/embedding_lookup/IdentityIdentity&embedding_30/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*9
_class/
-+loc:@embedding_30/embedding_lookup/41027208*,
_output_shapes
:??????????2(
&embedding_30/embedding_lookup/Identity?
(embedding_30/embedding_lookup/Identity_1Identity/embedding_30/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????2*
(embedding_30/embedding_lookup/Identity_1?
tf.cast_15/CastCast)tf.math.greater_equal_15/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2
tf.cast_15/Cast?
tf.__operators__.add_30/AddV2AddV21embedding_32/embedding_lookup/Identity_1:output:01embedding_30/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:??????????2
tf.__operators__.add_30/AddV2?
embedding_31/CastCast tf.math.floormod_10/FloorMod:z:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2
embedding_31/Cast?
embedding_31/embedding_lookupResourceGather&embedding_31_embedding_lookup_41027216embedding_31/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*9
_class/
-+loc:@embedding_31/embedding_lookup/41027216*,
_output_shapes
:??????????*
dtype02
embedding_31/embedding_lookup?
&embedding_31/embedding_lookup/IdentityIdentity&embedding_31/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*9
_class/
-+loc:@embedding_31/embedding_lookup/41027216*,
_output_shapes
:??????????2(
&embedding_31/embedding_lookup/Identity?
(embedding_31/embedding_lookup/Identity_1Identity/embedding_31/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????2*
(embedding_31/embedding_lookup/Identity_1?
tf.__operators__.add_31/AddV2AddV2!tf.__operators__.add_30/AddV2:z:01embedding_31/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:??????????2
tf.__operators__.add_31/AddV2?
 tf.expand_dims_10/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2"
 tf.expand_dims_10/ExpandDims/dim?
tf.expand_dims_10/ExpandDims
ExpandDimstf.cast_15/Cast:y:0)tf.expand_dims_10/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????2
tf.expand_dims_10/ExpandDims?
tf.math.multiply_10/MulMul!tf.__operators__.add_31/AddV2:z:0%tf.expand_dims_10/ExpandDims:output:0*
T0*,
_output_shapes
:??????????2
tf.math.multiply_10/Mul?
+tf.math.reduce_sum_10/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2-
+tf.math.reduce_sum_10/Sum/reduction_indices?
tf.math.reduce_sum_10/SumSumtf.math.multiply_10/Mul:z:04tf.math.reduce_sum_10/Sum/reduction_indices:output:0*
T0*(
_output_shapes
:??????????2
tf.math.reduce_sum_10/Sum?
IdentityIdentity"tf.math.reduce_sum_10/Sum:output:0^embedding_30/embedding_lookup^embedding_31/embedding_lookup^embedding_32/embedding_lookup*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:?????????: :::2>
embedding_30/embedding_lookupembedding_30/embedding_lookup2>
embedding_31/embedding_lookupembedding_31/embedding_lookup2>
embedding_32/embedding_lookupembedding_32/embedding_lookup:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: 
?
?
1__inference_custom_model_5_layer_call_fn_41027144

inputs_0_0

inputs_0_1
inputs_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCall
inputs_0_0
inputs_0_1inputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29*-
Tin&
$2"*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*<
_read_only_resource_inputs
	
 !*2
config_proto" 

CPU

GPU2 *0J 8? *U
fPRN
L__inference_custom_model_5_layer_call_and_return_conditional_losses_410265222
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????:?????????:?????????
: : :::: :::: : ::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
'
_output_shapes
:?????????
$
_user_specified_name
inputs/0/0:SO
'
_output_shapes
:?????????
$
_user_specified_name
inputs/0/1:QM
'
_output_shapes
:?????????

"
_user_specified_name
inputs/1:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?	
?
F__inference_dense_45_layer_call_and_return_conditional_losses_41027375

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
I__inference_normalize_5_layer_call_and_return_conditional_losses_41027536
x3
/normalization_5_reshape_readvariableop_resource5
1normalization_5_reshape_1_readvariableop_resource
identity??&normalization_5/Reshape/ReadVariableOp?(normalization_5/Reshape_1/ReadVariableOp?
&normalization_5/Reshape/ReadVariableOpReadVariableOp/normalization_5_reshape_readvariableop_resource*
_output_shapes	
:?*
dtype02(
&normalization_5/Reshape/ReadVariableOp?
normalization_5/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2
normalization_5/Reshape/shape?
normalization_5/ReshapeReshape.normalization_5/Reshape/ReadVariableOp:value:0&normalization_5/Reshape/shape:output:0*
T0*
_output_shapes
:	?2
normalization_5/Reshape?
(normalization_5/Reshape_1/ReadVariableOpReadVariableOp1normalization_5_reshape_1_readvariableop_resource*
_output_shapes	
:?*
dtype02*
(normalization_5/Reshape_1/ReadVariableOp?
normalization_5/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2!
normalization_5/Reshape_1/shape?
normalization_5/Reshape_1Reshape0normalization_5/Reshape_1/ReadVariableOp:value:0(normalization_5/Reshape_1/shape:output:0*
T0*
_output_shapes
:	?2
normalization_5/Reshape_1?
normalization_5/subSubx normalization_5/Reshape:output:0*
T0*(
_output_shapes
:??????????2
normalization_5/sub?
normalization_5/SqrtSqrt"normalization_5/Reshape_1:output:0*
T0*
_output_shapes
:	?2
normalization_5/Sqrt{
normalization_5/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32
normalization_5/Maximum/y?
normalization_5/MaximumMaximumnormalization_5/Sqrt:y:0"normalization_5/Maximum/y:output:0*
T0*
_output_shapes
:	?2
normalization_5/Maximum?
normalization_5/truedivRealDivnormalization_5/sub:z:0normalization_5/Maximum:z:0*
T0*(
_output_shapes
:??????????2
normalization_5/truediv?
IdentityIdentitynormalization_5/truediv:z:0'^normalization_5/Reshape/ReadVariableOp)^normalization_5/Reshape_1/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::2P
&normalization_5/Reshape/ReadVariableOp&normalization_5/Reshape/ReadVariableOp2T
(normalization_5/Reshape_1/ReadVariableOp(normalization_5/Reshape_1/ReadVariableOp:K G
(
_output_shapes
:??????????

_user_specified_namex
?	
?
J__inference_embedding_35_layer_call_and_return_conditional_losses_41025619

inputs
embedding_lookup_41025613
identity??embedding_lookup]
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:?????????2
Cast?
embedding_lookupResourceGatherembedding_lookup_41025613Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*,
_class"
 loc:@embedding_lookup/41025613*,
_output_shapes
:??????????*
dtype02
embedding_lookup?
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*,
_class"
 loc:@embedding_lookup/41025613*,
_output_shapes
:??????????2
embedding_lookup/Identity?
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????2
embedding_lookup/Identity_1?
IdentityIdentity$embedding_lookup/Identity_1:output:0^embedding_lookup*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????:2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
&__inference_signature_wrapper_41026666
bets

cards0

cards1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallcards0cards1betsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29*-
Tin&
$2"*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*<
_read_only_resource_inputs
	
 !*2
config_proto" 

CPU

GPU2 *0J 8? *,
f'R%
#__inference__wrapped_model_410253552
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????
:?????????:?????????: : :::: :::: : ::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:M I
'
_output_shapes
:?????????


_user_specified_namebets:OK
'
_output_shapes
:?????????
 
_user_specified_namecards0:OK
'
_output_shapes
:?????????
 
_user_specified_namecards1:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?	
?
F__inference_dense_47_layer_call_and_return_conditional_losses_41025985

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
5
bets-
serving_default_bets:0?????????

9
cards0/
serving_default_cards0:0?????????
9
cards1/
serving_default_cards1:0?????????<
dense_530
StatefulPartitionedCall:0?????????tensorflow/serving/predict:??
ۣ
layer-0
layer-1
layer-2
layer_with_weights-0
layer-3
layer_with_weights-1
layer-4
layer-5
layer-6
layer-7
	layer-8

layer_with_weights-2

layer-9
layer-10
layer_with_weights-3
layer-11
layer_with_weights-4
layer-12
layer_with_weights-5
layer-13
layer_with_weights-6
layer-14
layer-15
layer_with_weights-7
layer-16
layer-17
layer_with_weights-8
layer-18
layer-19
layer-20
layer_with_weights-9
layer-21
layer-22
layer-23
layer_with_weights-10
layer-24
layer_with_weights-11
layer-25
	optimizer
loss
trainable_variables
	variables
regularization_losses
 	keras_api
!
signatures
?_default_save_signature
?__call__
+?&call_and_return_all_conditional_losses"??
_tf_keras_networkɜ{"class_name": "CustomModel", "name": "custom_model_5", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "custom_model_5", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 2]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "cards0"}, "name": "cards0", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "cards1"}, "name": "cards1", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 10]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "bets"}, "name": "bets", "inbound_nodes": []}, {"class_name": "Functional", "config": {"name": "model_10", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 2]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_11"}, "name": "input_11", "inbound_nodes": []}, {"class_name": "Flatten", "config": {"name": "flatten_10", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_10", "inbound_nodes": [[["input_11", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.clip_by_value_15", "trainable": true, "dtype": "float32", "function": "clip_by_value"}, "name": "tf.clip_by_value_15", "inbound_nodes": [["flatten_10", 0, 0, {"clip_value_min": 0, "clip_value_max": 1000000.0}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.compat.v1.floor_div_10", "trainable": true, "dtype": "float32", "function": "compat.v1.floor_div"}, "name": "tf.compat.v1.floor_div_10", "inbound_nodes": [["tf.clip_by_value_15", 0, 0, {"y": 4, "name": null}]]}, {"class_name": "Embedding", "config": {"name": "embedding_32", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 52, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding_32", "inbound_nodes": [[["tf.clip_by_value_15", 0, 0, {}]]]}, {"class_name": "Embedding", "config": {"name": "embedding_30", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 13, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding_30", "inbound_nodes": [[["tf.compat.v1.floor_div_10", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.floormod_10", "trainable": true, "dtype": "float32", "function": "math.floormod"}, "name": "tf.math.floormod_10", "inbound_nodes": [["tf.clip_by_value_15", 0, 0, {"y": 4, "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.greater_equal_15", "trainable": true, "dtype": "float32", "function": "math.greater_equal"}, "name": "tf.math.greater_equal_15", "inbound_nodes": [["flatten_10", 0, 0, {"y": 0.0, "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_30", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_30", "inbound_nodes": [["embedding_32", 0, 0, {"y": ["embedding_30", 0, 0], "name": null}]]}, {"class_name": "Embedding", "config": {"name": "embedding_31", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 4, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding_31", "inbound_nodes": [[["tf.math.floormod_10", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.cast_15", "trainable": true, "dtype": "float32", "function": "cast"}, "name": "tf.cast_15", "inbound_nodes": [["tf.math.greater_equal_15", 0, 0, {"dtype": "float32"}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_31", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_31", "inbound_nodes": [["tf.__operators__.add_30", 0, 0, {"y": ["embedding_31", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.expand_dims_10", "trainable": true, "dtype": "float32", "function": "expand_dims"}, "name": "tf.expand_dims_10", "inbound_nodes": [["tf.cast_15", 0, 0, {"axis": -1}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.multiply_10", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "name": "tf.math.multiply_10", "inbound_nodes": [["tf.__operators__.add_31", 0, 0, {"y": ["tf.expand_dims_10", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.reduce_sum_10", "trainable": true, "dtype": "float32", "function": "math.reduce_sum"}, "name": "tf.math.reduce_sum_10", "inbound_nodes": [["tf.math.multiply_10", 0, 0, {"axis": 1}]]}], "input_layers": [["input_11", 0, 0]], "output_layers": [["tf.math.reduce_sum_10", 0, 0]]}, "name": "model_10", "inbound_nodes": [[["cards0", 0, 0, {}]]]}, {"class_name": "Functional", "config": {"name": "model_11", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_12"}, "name": "input_12", "inbound_nodes": []}, {"class_name": "Flatten", "config": {"name": "flatten_11", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_11", "inbound_nodes": [[["input_12", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.clip_by_value_16", "trainable": true, "dtype": "float32", "function": "clip_by_value"}, "name": "tf.clip_by_value_16", "inbound_nodes": [["flatten_11", 0, 0, {"clip_value_min": 0, "clip_value_max": 1000000.0}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.compat.v1.floor_div_11", "trainable": true, "dtype": "float32", "function": "compat.v1.floor_div"}, "name": "tf.compat.v1.floor_div_11", "inbound_nodes": [["tf.clip_by_value_16", 0, 0, {"y": 4, "name": null}]]}, {"class_name": "Embedding", "config": {"name": "embedding_35", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 52, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding_35", "inbound_nodes": [[["tf.clip_by_value_16", 0, 0, {}]]]}, {"class_name": "Embedding", "config": {"name": "embedding_33", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 13, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding_33", "inbound_nodes": [[["tf.compat.v1.floor_div_11", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.floormod_11", "trainable": true, "dtype": "float32", "function": "math.floormod"}, "name": "tf.math.floormod_11", "inbound_nodes": [["tf.clip_by_value_16", 0, 0, {"y": 4, "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.greater_equal_16", "trainable": true, "dtype": "float32", "function": "math.greater_equal"}, "name": "tf.math.greater_equal_16", "inbound_nodes": [["flatten_11", 0, 0, {"y": 0.0, "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_32", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_32", "inbound_nodes": [["embedding_35", 0, 0, {"y": ["embedding_33", 0, 0], "name": null}]]}, {"class_name": "Embedding", "config": {"name": "embedding_34", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 4, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding_34", "inbound_nodes": [[["tf.math.floormod_11", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.cast_16", "trainable": true, "dtype": "float32", "function": "cast"}, "name": "tf.cast_16", "inbound_nodes": [["tf.math.greater_equal_16", 0, 0, {"dtype": "float32"}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_33", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_33", "inbound_nodes": [["tf.__operators__.add_32", 0, 0, {"y": ["embedding_34", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.expand_dims_11", "trainable": true, "dtype": "float32", "function": "expand_dims"}, "name": "tf.expand_dims_11", "inbound_nodes": [["tf.cast_16", 0, 0, {"axis": -1}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.multiply_11", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "name": "tf.math.multiply_11", "inbound_nodes": [["tf.__operators__.add_33", 0, 0, {"y": ["tf.expand_dims_11", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.reduce_sum_11", "trainable": true, "dtype": "float32", "function": "math.reduce_sum"}, "name": "tf.math.reduce_sum_11", "inbound_nodes": [["tf.math.multiply_11", 0, 0, {"axis": 1}]]}], "input_layers": [["input_12", 0, 0]], "output_layers": [["tf.math.reduce_sum_11", 0, 0]]}, "name": "model_11", "inbound_nodes": [[["cards1", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.greater_equal_17", "trainable": true, "dtype": "float32", "function": "math.greater_equal"}, "name": "tf.math.greater_equal_17", "inbound_nodes": [["bets", 0, 0, {"y": 0.0, "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.concat_15", "trainable": true, "dtype": "float32", "function": "concat"}, "name": "tf.concat_15", "inbound_nodes": [[["model_10", 1, 0, {"axis": 1}], ["model_11", 1, 0, {"axis": 1}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.clip_by_value_17", "trainable": true, "dtype": "float32", "function": "clip_by_value"}, "name": "tf.clip_by_value_17", "inbound_nodes": [["bets", 0, 0, {"clip_value_min": 0.0, "clip_value_max": 1000000.0}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.cast_17", "trainable": true, "dtype": "float32", "function": "cast"}, "name": "tf.cast_17", "inbound_nodes": [["tf.math.greater_equal_17", 0, 0, {"dtype": "float32"}]]}, {"class_name": "Dense", "config": {"name": "dense_45", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_45", "inbound_nodes": [[["tf.concat_15", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.concat_16", "trainable": true, "dtype": "float32", "function": "concat"}, "name": "tf.concat_16", "inbound_nodes": [[["tf.clip_by_value_17", 0, 0, {"axis": -1}], ["tf.cast_17", 0, 0, {"axis": -1}]]]}, {"class_name": "Dense", "config": {"name": "dense_46", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_46", "inbound_nodes": [[["dense_45", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_48", "trainable": true, "dtype": "float32", "units": 256, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_48", "inbound_nodes": [[["tf.concat_16", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_47", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_47", "inbound_nodes": [[["dense_46", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_49", "trainable": true, "dtype": "float32", "units": 256, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_49", "inbound_nodes": [[["dense_48", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.concat_17", "trainable": true, "dtype": "float32", "function": "concat"}, "name": "tf.concat_17", "inbound_nodes": [[["dense_47", 0, 0, {"axis": -1}], ["dense_49", 0, 0, {"axis": -1}]]]}, {"class_name": "Dense", "config": {"name": "dense_50", "trainable": true, "dtype": "float32", "units": 256, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_50", "inbound_nodes": [[["tf.concat_17", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.nn.relu_15", "trainable": true, "dtype": "float32", "function": "nn.relu"}, "name": "tf.nn.relu_15", "inbound_nodes": [["dense_50", 0, 0, {"name": null}]]}, {"class_name": "Dense", "config": {"name": "dense_51", "trainable": true, "dtype": "float32", "units": 256, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_51", "inbound_nodes": [[["tf.nn.relu_15", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_34", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_34", "inbound_nodes": [["dense_51", 0, 0, {"y": ["tf.nn.relu_15", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.nn.relu_16", "trainable": true, "dtype": "float32", "function": "nn.relu"}, "name": "tf.nn.relu_16", "inbound_nodes": [["tf.__operators__.add_34", 0, 0, {"name": null}]]}, {"class_name": "Dense", "config": {"name": "dense_52", "trainable": true, "dtype": "float32", "units": 256, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_52", "inbound_nodes": [[["tf.nn.relu_16", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_35", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_35", "inbound_nodes": [["dense_52", 0, 0, {"y": ["tf.nn.relu_16", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.nn.relu_17", "trainable": true, "dtype": "float32", "function": "nn.relu"}, "name": "tf.nn.relu_17", "inbound_nodes": [["tf.__operators__.add_35", 0, 0, {"name": null}]]}, {"class_name": "Normalize", "config": {"name": "normalize_5", "trainable": true, "dtype": "float32"}, "name": "normalize_5", "inbound_nodes": [[["tf.nn.relu_17", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_53", "trainable": true, "dtype": "float32", "units": 4, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Constant", "config": {"value": 0}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_53", "inbound_nodes": [[["normalize_5", 0, 0, {}]]]}], "input_layers": [[["cards0", 0, 0], ["cards1", 0, 0]], ["bets", 0, 0]], "output_layers": [["dense_53", 0, 0]]}, "build_input_shape": [[{"class_name": "TensorShape", "items": [null, 2]}, {"class_name": "TensorShape", "items": [null, 3]}], {"class_name": "TensorShape", "items": [null, 10]}], "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "CustomModel", "config": {"name": "custom_model_5", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 2]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "cards0"}, "name": "cards0", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "cards1"}, "name": "cards1", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 10]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "bets"}, "name": "bets", "inbound_nodes": []}, {"class_name": "Functional", "config": {"name": "model_10", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 2]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_11"}, "name": "input_11", "inbound_nodes": []}, {"class_name": "Flatten", "config": {"name": "flatten_10", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_10", "inbound_nodes": [[["input_11", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.clip_by_value_15", "trainable": true, "dtype": "float32", "function": "clip_by_value"}, "name": "tf.clip_by_value_15", "inbound_nodes": [["flatten_10", 0, 0, {"clip_value_min": 0, "clip_value_max": 1000000.0}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.compat.v1.floor_div_10", "trainable": true, "dtype": "float32", "function": "compat.v1.floor_div"}, "name": "tf.compat.v1.floor_div_10", "inbound_nodes": [["tf.clip_by_value_15", 0, 0, {"y": 4, "name": null}]]}, {"class_name": "Embedding", "config": {"name": "embedding_32", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 52, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding_32", "inbound_nodes": [[["tf.clip_by_value_15", 0, 0, {}]]]}, {"class_name": "Embedding", "config": {"name": "embedding_30", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 13, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding_30", "inbound_nodes": [[["tf.compat.v1.floor_div_10", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.floormod_10", "trainable": true, "dtype": "float32", "function": "math.floormod"}, "name": "tf.math.floormod_10", "inbound_nodes": [["tf.clip_by_value_15", 0, 0, {"y": 4, "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.greater_equal_15", "trainable": true, "dtype": "float32", "function": "math.greater_equal"}, "name": "tf.math.greater_equal_15", "inbound_nodes": [["flatten_10", 0, 0, {"y": 0.0, "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_30", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_30", "inbound_nodes": [["embedding_32", 0, 0, {"y": ["embedding_30", 0, 0], "name": null}]]}, {"class_name": "Embedding", "config": {"name": "embedding_31", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 4, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding_31", "inbound_nodes": [[["tf.math.floormod_10", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.cast_15", "trainable": true, "dtype": "float32", "function": "cast"}, "name": "tf.cast_15", "inbound_nodes": [["tf.math.greater_equal_15", 0, 0, {"dtype": "float32"}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_31", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_31", "inbound_nodes": [["tf.__operators__.add_30", 0, 0, {"y": ["embedding_31", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.expand_dims_10", "trainable": true, "dtype": "float32", "function": "expand_dims"}, "name": "tf.expand_dims_10", "inbound_nodes": [["tf.cast_15", 0, 0, {"axis": -1}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.multiply_10", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "name": "tf.math.multiply_10", "inbound_nodes": [["tf.__operators__.add_31", 0, 0, {"y": ["tf.expand_dims_10", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.reduce_sum_10", "trainable": true, "dtype": "float32", "function": "math.reduce_sum"}, "name": "tf.math.reduce_sum_10", "inbound_nodes": [["tf.math.multiply_10", 0, 0, {"axis": 1}]]}], "input_layers": [["input_11", 0, 0]], "output_layers": [["tf.math.reduce_sum_10", 0, 0]]}, "name": "model_10", "inbound_nodes": [[["cards0", 0, 0, {}]]]}, {"class_name": "Functional", "config": {"name": "model_11", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_12"}, "name": "input_12", "inbound_nodes": []}, {"class_name": "Flatten", "config": {"name": "flatten_11", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_11", "inbound_nodes": [[["input_12", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.clip_by_value_16", "trainable": true, "dtype": "float32", "function": "clip_by_value"}, "name": "tf.clip_by_value_16", "inbound_nodes": [["flatten_11", 0, 0, {"clip_value_min": 0, "clip_value_max": 1000000.0}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.compat.v1.floor_div_11", "trainable": true, "dtype": "float32", "function": "compat.v1.floor_div"}, "name": "tf.compat.v1.floor_div_11", "inbound_nodes": [["tf.clip_by_value_16", 0, 0, {"y": 4, "name": null}]]}, {"class_name": "Embedding", "config": {"name": "embedding_35", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 52, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding_35", "inbound_nodes": [[["tf.clip_by_value_16", 0, 0, {}]]]}, {"class_name": "Embedding", "config": {"name": "embedding_33", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 13, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding_33", "inbound_nodes": [[["tf.compat.v1.floor_div_11", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.floormod_11", "trainable": true, "dtype": "float32", "function": "math.floormod"}, "name": "tf.math.floormod_11", "inbound_nodes": [["tf.clip_by_value_16", 0, 0, {"y": 4, "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.greater_equal_16", "trainable": true, "dtype": "float32", "function": "math.greater_equal"}, "name": "tf.math.greater_equal_16", "inbound_nodes": [["flatten_11", 0, 0, {"y": 0.0, "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_32", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_32", "inbound_nodes": [["embedding_35", 0, 0, {"y": ["embedding_33", 0, 0], "name": null}]]}, {"class_name": "Embedding", "config": {"name": "embedding_34", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 4, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding_34", "inbound_nodes": [[["tf.math.floormod_11", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.cast_16", "trainable": true, "dtype": "float32", "function": "cast"}, "name": "tf.cast_16", "inbound_nodes": [["tf.math.greater_equal_16", 0, 0, {"dtype": "float32"}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_33", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_33", "inbound_nodes": [["tf.__operators__.add_32", 0, 0, {"y": ["embedding_34", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.expand_dims_11", "trainable": true, "dtype": "float32", "function": "expand_dims"}, "name": "tf.expand_dims_11", "inbound_nodes": [["tf.cast_16", 0, 0, {"axis": -1}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.multiply_11", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "name": "tf.math.multiply_11", "inbound_nodes": [["tf.__operators__.add_33", 0, 0, {"y": ["tf.expand_dims_11", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.reduce_sum_11", "trainable": true, "dtype": "float32", "function": "math.reduce_sum"}, "name": "tf.math.reduce_sum_11", "inbound_nodes": [["tf.math.multiply_11", 0, 0, {"axis": 1}]]}], "input_layers": [["input_12", 0, 0]], "output_layers": [["tf.math.reduce_sum_11", 0, 0]]}, "name": "model_11", "inbound_nodes": [[["cards1", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.greater_equal_17", "trainable": true, "dtype": "float32", "function": "math.greater_equal"}, "name": "tf.math.greater_equal_17", "inbound_nodes": [["bets", 0, 0, {"y": 0.0, "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.concat_15", "trainable": true, "dtype": "float32", "function": "concat"}, "name": "tf.concat_15", "inbound_nodes": [[["model_10", 1, 0, {"axis": 1}], ["model_11", 1, 0, {"axis": 1}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.clip_by_value_17", "trainable": true, "dtype": "float32", "function": "clip_by_value"}, "name": "tf.clip_by_value_17", "inbound_nodes": [["bets", 0, 0, {"clip_value_min": 0.0, "clip_value_max": 1000000.0}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.cast_17", "trainable": true, "dtype": "float32", "function": "cast"}, "name": "tf.cast_17", "inbound_nodes": [["tf.math.greater_equal_17", 0, 0, {"dtype": "float32"}]]}, {"class_name": "Dense", "config": {"name": "dense_45", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_45", "inbound_nodes": [[["tf.concat_15", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.concat_16", "trainable": true, "dtype": "float32", "function": "concat"}, "name": "tf.concat_16", "inbound_nodes": [[["tf.clip_by_value_17", 0, 0, {"axis": -1}], ["tf.cast_17", 0, 0, {"axis": -1}]]]}, {"class_name": "Dense", "config": {"name": "dense_46", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_46", "inbound_nodes": [[["dense_45", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_48", "trainable": true, "dtype": "float32", "units": 256, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_48", "inbound_nodes": [[["tf.concat_16", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_47", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_47", "inbound_nodes": [[["dense_46", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_49", "trainable": true, "dtype": "float32", "units": 256, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_49", "inbound_nodes": [[["dense_48", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.concat_17", "trainable": true, "dtype": "float32", "function": "concat"}, "name": "tf.concat_17", "inbound_nodes": [[["dense_47", 0, 0, {"axis": -1}], ["dense_49", 0, 0, {"axis": -1}]]]}, {"class_name": "Dense", "config": {"name": "dense_50", "trainable": true, "dtype": "float32", "units": 256, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_50", "inbound_nodes": [[["tf.concat_17", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.nn.relu_15", "trainable": true, "dtype": "float32", "function": "nn.relu"}, "name": "tf.nn.relu_15", "inbound_nodes": [["dense_50", 0, 0, {"name": null}]]}, {"class_name": "Dense", "config": {"name": "dense_51", "trainable": true, "dtype": "float32", "units": 256, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_51", "inbound_nodes": [[["tf.nn.relu_15", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_34", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_34", "inbound_nodes": [["dense_51", 0, 0, {"y": ["tf.nn.relu_15", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.nn.relu_16", "trainable": true, "dtype": "float32", "function": "nn.relu"}, "name": "tf.nn.relu_16", "inbound_nodes": [["tf.__operators__.add_34", 0, 0, {"name": null}]]}, {"class_name": "Dense", "config": {"name": "dense_52", "trainable": true, "dtype": "float32", "units": 256, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_52", "inbound_nodes": [[["tf.nn.relu_16", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_35", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_35", "inbound_nodes": [["dense_52", 0, 0, {"y": ["tf.nn.relu_16", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.nn.relu_17", "trainable": true, "dtype": "float32", "function": "nn.relu"}, "name": "tf.nn.relu_17", "inbound_nodes": [["tf.__operators__.add_35", 0, 0, {"name": null}]]}, {"class_name": "Normalize", "config": {"name": "normalize_5", "trainable": true, "dtype": "float32"}, "name": "normalize_5", "inbound_nodes": [[["tf.nn.relu_17", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_53", "trainable": true, "dtype": "float32", "units": 4, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Constant", "config": {"value": 0}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_53", "inbound_nodes": [[["normalize_5", 0, 0, {}]]]}], "input_layers": [[["cards0", 0, 0], ["cards1", 0, 0]], ["bets", 0, 0]], "output_layers": [["dense_53", 0, 0]]}}, "training_config": {"loss": null, "metrics": null, "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "cards0", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 2]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 2]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "cards0"}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "cards1", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 3]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "cards1"}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "bets", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 10]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 10]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "bets"}}
?R
"layer-0
#layer-1
$layer-2
%layer-3
&layer_with_weights-0
&layer-4
'layer_with_weights-1
'layer-5
(layer-6
)layer-7
*layer-8
+layer_with_weights-2
+layer-9
,layer-10
-layer-11
.layer-12
/layer-13
0layer-14
1trainable_variables
2	variables
3regularization_losses
4	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?N
_tf_keras_network?N{"class_name": "Functional", "name": "model_10", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "model_10", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 2]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_11"}, "name": "input_11", "inbound_nodes": []}, {"class_name": "Flatten", "config": {"name": "flatten_10", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_10", "inbound_nodes": [[["input_11", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.clip_by_value_15", "trainable": true, "dtype": "float32", "function": "clip_by_value"}, "name": "tf.clip_by_value_15", "inbound_nodes": [["flatten_10", 0, 0, {"clip_value_min": 0, "clip_value_max": 1000000.0}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.compat.v1.floor_div_10", "trainable": true, "dtype": "float32", "function": "compat.v1.floor_div"}, "name": "tf.compat.v1.floor_div_10", "inbound_nodes": [["tf.clip_by_value_15", 0, 0, {"y": 4, "name": null}]]}, {"class_name": "Embedding", "config": {"name": "embedding_32", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 52, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding_32", "inbound_nodes": [[["tf.clip_by_value_15", 0, 0, {}]]]}, {"class_name": "Embedding", "config": {"name": "embedding_30", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 13, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding_30", "inbound_nodes": [[["tf.compat.v1.floor_div_10", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.floormod_10", "trainable": true, "dtype": "float32", "function": "math.floormod"}, "name": "tf.math.floormod_10", "inbound_nodes": [["tf.clip_by_value_15", 0, 0, {"y": 4, "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.greater_equal_15", "trainable": true, "dtype": "float32", "function": "math.greater_equal"}, "name": "tf.math.greater_equal_15", "inbound_nodes": [["flatten_10", 0, 0, {"y": 0.0, "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_30", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_30", "inbound_nodes": [["embedding_32", 0, 0, {"y": ["embedding_30", 0, 0], "name": null}]]}, {"class_name": "Embedding", "config": {"name": "embedding_31", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 4, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding_31", "inbound_nodes": [[["tf.math.floormod_10", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.cast_15", "trainable": true, "dtype": "float32", "function": "cast"}, "name": "tf.cast_15", "inbound_nodes": [["tf.math.greater_equal_15", 0, 0, {"dtype": "float32"}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_31", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_31", "inbound_nodes": [["tf.__operators__.add_30", 0, 0, {"y": ["embedding_31", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.expand_dims_10", "trainable": true, "dtype": "float32", "function": "expand_dims"}, "name": "tf.expand_dims_10", "inbound_nodes": [["tf.cast_15", 0, 0, {"axis": -1}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.multiply_10", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "name": "tf.math.multiply_10", "inbound_nodes": [["tf.__operators__.add_31", 0, 0, {"y": ["tf.expand_dims_10", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.reduce_sum_10", "trainable": true, "dtype": "float32", "function": "math.reduce_sum"}, "name": "tf.math.reduce_sum_10", "inbound_nodes": [["tf.math.multiply_10", 0, 0, {"axis": 1}]]}], "input_layers": [["input_11", 0, 0]], "output_layers": [["tf.math.reduce_sum_10", 0, 0]]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 2]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 2]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "model_10", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 2]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_11"}, "name": "input_11", "inbound_nodes": []}, {"class_name": "Flatten", "config": {"name": "flatten_10", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_10", "inbound_nodes": [[["input_11", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.clip_by_value_15", "trainable": true, "dtype": "float32", "function": "clip_by_value"}, "name": "tf.clip_by_value_15", "inbound_nodes": [["flatten_10", 0, 0, {"clip_value_min": 0, "clip_value_max": 1000000.0}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.compat.v1.floor_div_10", "trainable": true, "dtype": "float32", "function": "compat.v1.floor_div"}, "name": "tf.compat.v1.floor_div_10", "inbound_nodes": [["tf.clip_by_value_15", 0, 0, {"y": 4, "name": null}]]}, {"class_name": "Embedding", "config": {"name": "embedding_32", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 52, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding_32", "inbound_nodes": [[["tf.clip_by_value_15", 0, 0, {}]]]}, {"class_name": "Embedding", "config": {"name": "embedding_30", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 13, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding_30", "inbound_nodes": [[["tf.compat.v1.floor_div_10", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.floormod_10", "trainable": true, "dtype": "float32", "function": "math.floormod"}, "name": "tf.math.floormod_10", "inbound_nodes": [["tf.clip_by_value_15", 0, 0, {"y": 4, "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.greater_equal_15", "trainable": true, "dtype": "float32", "function": "math.greater_equal"}, "name": "tf.math.greater_equal_15", "inbound_nodes": [["flatten_10", 0, 0, {"y": 0.0, "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_30", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_30", "inbound_nodes": [["embedding_32", 0, 0, {"y": ["embedding_30", 0, 0], "name": null}]]}, {"class_name": "Embedding", "config": {"name": "embedding_31", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 4, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding_31", "inbound_nodes": [[["tf.math.floormod_10", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.cast_15", "trainable": true, "dtype": "float32", "function": "cast"}, "name": "tf.cast_15", "inbound_nodes": [["tf.math.greater_equal_15", 0, 0, {"dtype": "float32"}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_31", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_31", "inbound_nodes": [["tf.__operators__.add_30", 0, 0, {"y": ["embedding_31", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.expand_dims_10", "trainable": true, "dtype": "float32", "function": "expand_dims"}, "name": "tf.expand_dims_10", "inbound_nodes": [["tf.cast_15", 0, 0, {"axis": -1}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.multiply_10", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "name": "tf.math.multiply_10", "inbound_nodes": [["tf.__operators__.add_31", 0, 0, {"y": ["tf.expand_dims_10", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.reduce_sum_10", "trainable": true, "dtype": "float32", "function": "math.reduce_sum"}, "name": "tf.math.reduce_sum_10", "inbound_nodes": [["tf.math.multiply_10", 0, 0, {"axis": 1}]]}], "input_layers": [["input_11", 0, 0]], "output_layers": [["tf.math.reduce_sum_10", 0, 0]]}}}
?R
5layer-0
6layer-1
7layer-2
8layer-3
9layer_with_weights-0
9layer-4
:layer_with_weights-1
:layer-5
;layer-6
<layer-7
=layer-8
>layer_with_weights-2
>layer-9
?layer-10
@layer-11
Alayer-12
Blayer-13
Clayer-14
Dtrainable_variables
E	variables
Fregularization_losses
G	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?N
_tf_keras_network?N{"class_name": "Functional", "name": "model_11", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "model_11", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_12"}, "name": "input_12", "inbound_nodes": []}, {"class_name": "Flatten", "config": {"name": "flatten_11", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_11", "inbound_nodes": [[["input_12", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.clip_by_value_16", "trainable": true, "dtype": "float32", "function": "clip_by_value"}, "name": "tf.clip_by_value_16", "inbound_nodes": [["flatten_11", 0, 0, {"clip_value_min": 0, "clip_value_max": 1000000.0}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.compat.v1.floor_div_11", "trainable": true, "dtype": "float32", "function": "compat.v1.floor_div"}, "name": "tf.compat.v1.floor_div_11", "inbound_nodes": [["tf.clip_by_value_16", 0, 0, {"y": 4, "name": null}]]}, {"class_name": "Embedding", "config": {"name": "embedding_35", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 52, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding_35", "inbound_nodes": [[["tf.clip_by_value_16", 0, 0, {}]]]}, {"class_name": "Embedding", "config": {"name": "embedding_33", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 13, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding_33", "inbound_nodes": [[["tf.compat.v1.floor_div_11", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.floormod_11", "trainable": true, "dtype": "float32", "function": "math.floormod"}, "name": "tf.math.floormod_11", "inbound_nodes": [["tf.clip_by_value_16", 0, 0, {"y": 4, "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.greater_equal_16", "trainable": true, "dtype": "float32", "function": "math.greater_equal"}, "name": "tf.math.greater_equal_16", "inbound_nodes": [["flatten_11", 0, 0, {"y": 0.0, "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_32", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_32", "inbound_nodes": [["embedding_35", 0, 0, {"y": ["embedding_33", 0, 0], "name": null}]]}, {"class_name": "Embedding", "config": {"name": "embedding_34", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 4, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding_34", "inbound_nodes": [[["tf.math.floormod_11", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.cast_16", "trainable": true, "dtype": "float32", "function": "cast"}, "name": "tf.cast_16", "inbound_nodes": [["tf.math.greater_equal_16", 0, 0, {"dtype": "float32"}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_33", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_33", "inbound_nodes": [["tf.__operators__.add_32", 0, 0, {"y": ["embedding_34", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.expand_dims_11", "trainable": true, "dtype": "float32", "function": "expand_dims"}, "name": "tf.expand_dims_11", "inbound_nodes": [["tf.cast_16", 0, 0, {"axis": -1}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.multiply_11", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "name": "tf.math.multiply_11", "inbound_nodes": [["tf.__operators__.add_33", 0, 0, {"y": ["tf.expand_dims_11", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.reduce_sum_11", "trainable": true, "dtype": "float32", "function": "math.reduce_sum"}, "name": "tf.math.reduce_sum_11", "inbound_nodes": [["tf.math.multiply_11", 0, 0, {"axis": 1}]]}], "input_layers": [["input_12", 0, 0]], "output_layers": [["tf.math.reduce_sum_11", 0, 0]]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 3]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 3]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "model_11", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_12"}, "name": "input_12", "inbound_nodes": []}, {"class_name": "Flatten", "config": {"name": "flatten_11", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_11", "inbound_nodes": [[["input_12", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.clip_by_value_16", "trainable": true, "dtype": "float32", "function": "clip_by_value"}, "name": "tf.clip_by_value_16", "inbound_nodes": [["flatten_11", 0, 0, {"clip_value_min": 0, "clip_value_max": 1000000.0}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.compat.v1.floor_div_11", "trainable": true, "dtype": "float32", "function": "compat.v1.floor_div"}, "name": "tf.compat.v1.floor_div_11", "inbound_nodes": [["tf.clip_by_value_16", 0, 0, {"y": 4, "name": null}]]}, {"class_name": "Embedding", "config": {"name": "embedding_35", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 52, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding_35", "inbound_nodes": [[["tf.clip_by_value_16", 0, 0, {}]]]}, {"class_name": "Embedding", "config": {"name": "embedding_33", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 13, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding_33", "inbound_nodes": [[["tf.compat.v1.floor_div_11", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.floormod_11", "trainable": true, "dtype": "float32", "function": "math.floormod"}, "name": "tf.math.floormod_11", "inbound_nodes": [["tf.clip_by_value_16", 0, 0, {"y": 4, "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.greater_equal_16", "trainable": true, "dtype": "float32", "function": "math.greater_equal"}, "name": "tf.math.greater_equal_16", "inbound_nodes": [["flatten_11", 0, 0, {"y": 0.0, "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_32", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_32", "inbound_nodes": [["embedding_35", 0, 0, {"y": ["embedding_33", 0, 0], "name": null}]]}, {"class_name": "Embedding", "config": {"name": "embedding_34", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 4, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding_34", "inbound_nodes": [[["tf.math.floormod_11", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.cast_16", "trainable": true, "dtype": "float32", "function": "cast"}, "name": "tf.cast_16", "inbound_nodes": [["tf.math.greater_equal_16", 0, 0, {"dtype": "float32"}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_33", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_33", "inbound_nodes": [["tf.__operators__.add_32", 0, 0, {"y": ["embedding_34", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.expand_dims_11", "trainable": true, "dtype": "float32", "function": "expand_dims"}, "name": "tf.expand_dims_11", "inbound_nodes": [["tf.cast_16", 0, 0, {"axis": -1}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.multiply_11", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "name": "tf.math.multiply_11", "inbound_nodes": [["tf.__operators__.add_33", 0, 0, {"y": ["tf.expand_dims_11", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.reduce_sum_11", "trainable": true, "dtype": "float32", "function": "math.reduce_sum"}, "name": "tf.math.reduce_sum_11", "inbound_nodes": [["tf.math.multiply_11", 0, 0, {"axis": 1}]]}], "input_layers": [["input_12", 0, 0]], "output_layers": [["tf.math.reduce_sum_11", 0, 0]]}}}
?
H	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.math.greater_equal_17", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.math.greater_equal_17", "trainable": true, "dtype": "float32", "function": "math.greater_equal"}}
?
I	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.concat_15", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.concat_15", "trainable": true, "dtype": "float32", "function": "concat"}}
?
J	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.clip_by_value_17", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.clip_by_value_17", "trainable": true, "dtype": "float32", "function": "clip_by_value"}}
?
K	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.cast_17", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.cast_17", "trainable": true, "dtype": "float32", "function": "cast"}}
?

Lkernel
Mbias
Ntrainable_variables
O	variables
Pregularization_losses
Q	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_45", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_45", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 512}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 512]}}
?
R	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.concat_16", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.concat_16", "trainable": true, "dtype": "float32", "function": "concat"}}
?

Skernel
Tbias
Utrainable_variables
V	variables
Wregularization_losses
X	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_46", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_46", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 256]}}
?

Ykernel
Zbias
[trainable_variables
\	variables
]regularization_losses
^	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_48", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_48", "trainable": true, "dtype": "float32", "units": 256, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 20}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 20]}}
?

_kernel
`bias
atrainable_variables
b	variables
cregularization_losses
d	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_47", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_47", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 256]}}
?

ekernel
fbias
gtrainable_variables
h	variables
iregularization_losses
j	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_49", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_49", "trainable": true, "dtype": "float32", "units": 256, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 256]}}
?
k	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.concat_17", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.concat_17", "trainable": true, "dtype": "float32", "function": "concat"}}
?

lkernel
mbias
ntrainable_variables
o	variables
pregularization_losses
q	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_50", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_50", "trainable": true, "dtype": "float32", "units": 256, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 512}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 512]}}
?
r	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.nn.relu_15", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.nn.relu_15", "trainable": true, "dtype": "float32", "function": "nn.relu"}}
?

skernel
tbias
utrainable_variables
v	variables
wregularization_losses
x	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_51", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_51", "trainable": true, "dtype": "float32", "units": 256, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 256]}}
?
y	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.__operators__.add_34", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.__operators__.add_34", "trainable": true, "dtype": "float32", "function": "__operators__.add"}}
?
z	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.nn.relu_16", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.nn.relu_16", "trainable": true, "dtype": "float32", "function": "nn.relu"}}
?

{kernel
|bias
}trainable_variables
~	variables
regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_52", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_52", "trainable": true, "dtype": "float32", "units": 256, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 256]}}
?
?	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.__operators__.add_35", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.__operators__.add_35", "trainable": true, "dtype": "float32", "function": "__operators__.add"}}
?
?	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.nn.relu_17", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.nn.relu_17", "trainable": true, "dtype": "float32", "function": "nn.relu"}}
?
?	normalize
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Normalize", "name": "normalize_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "normalize_5", "trainable": true, "dtype": "float32"}}
?
?kernel
	?bias
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_53", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_53", "trainable": true, "dtype": "float32", "units": 4, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Constant", "config": {"value": 0}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 256]}}
?
	?iter
?beta_1
?beta_2

?decay
?learning_rateLm?Mm?Sm?Tm?Ym?Zm?_m?`m?em?fm?lm?mm?sm?tm?{m?|m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?Lv?Mv?Sv?Tv?Yv?Zv?_v?`v?ev?fv?lv?mv?sv?tv?{v?|v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?"
	optimizer
 "
trackable_dict_wrapper
?
?0
?1
?2
?3
?4
?5
L6
M7
S8
T9
Y10
Z11
_12
`13
e14
f15
l16
m17
s18
t19
{20
|21
?22
?23"
trackable_list_wrapper
?
?0
?1
?2
?3
?4
?5
L6
M7
S8
T9
Y10
Z11
_12
`13
e14
f15
l16
m17
s18
t19
{20
|21
?22
?23
?24
?25
?26"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
trainable_variables
 ?layer_regularization_losses
	variables
regularization_losses
?layer_metrics
?metrics
?non_trainable_variables
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "input_11", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 2]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 2]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_11"}}
?
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Flatten", "name": "flatten_10", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten_10", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
?
?	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.clip_by_value_15", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.clip_by_value_15", "trainable": true, "dtype": "float32", "function": "clip_by_value"}}
?
?	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.compat.v1.floor_div_10", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.compat.v1.floor_div_10", "trainable": true, "dtype": "float32", "function": "compat.v1.floor_div"}}
?
?
embeddings
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Embedding", "name": "embedding_32", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "embedding_32", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 52, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 2]}}
?
?
embeddings
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Embedding", "name": "embedding_30", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "embedding_30", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 13, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 2]}}
?
?	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.math.floormod_10", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.math.floormod_10", "trainable": true, "dtype": "float32", "function": "math.floormod"}}
?
?	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.math.greater_equal_15", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.math.greater_equal_15", "trainable": true, "dtype": "float32", "function": "math.greater_equal"}}
?
?	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.__operators__.add_30", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.__operators__.add_30", "trainable": true, "dtype": "float32", "function": "__operators__.add"}}
?
?
embeddings
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Embedding", "name": "embedding_31", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "embedding_31", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 4, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 2]}}
?
?	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.cast_15", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.cast_15", "trainable": true, "dtype": "float32", "function": "cast"}}
?
?	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.__operators__.add_31", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.__operators__.add_31", "trainable": true, "dtype": "float32", "function": "__operators__.add"}}
?
?	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.expand_dims_10", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.expand_dims_10", "trainable": true, "dtype": "float32", "function": "expand_dims"}}
?
?	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.math.multiply_10", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.math.multiply_10", "trainable": true, "dtype": "float32", "function": "math.multiply"}}
?
?	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.math.reduce_sum_10", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.math.reduce_sum_10", "trainable": true, "dtype": "float32", "function": "math.reduce_sum"}}
8
?0
?1
?2"
trackable_list_wrapper
8
?0
?1
?2"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
1trainable_variables
 ?layer_regularization_losses
2	variables
3regularization_losses
?layer_metrics
?metrics
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "input_12", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 3]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_12"}}
?
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Flatten", "name": "flatten_11", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten_11", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
?
?	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.clip_by_value_16", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.clip_by_value_16", "trainable": true, "dtype": "float32", "function": "clip_by_value"}}
?
?	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.compat.v1.floor_div_11", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.compat.v1.floor_div_11", "trainable": true, "dtype": "float32", "function": "compat.v1.floor_div"}}
?
?
embeddings
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Embedding", "name": "embedding_35", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "embedding_35", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 52, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 3]}}
?
?
embeddings
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Embedding", "name": "embedding_33", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "embedding_33", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 13, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 3]}}
?
?	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.math.floormod_11", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.math.floormod_11", "trainable": true, "dtype": "float32", "function": "math.floormod"}}
?
?	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.math.greater_equal_16", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.math.greater_equal_16", "trainable": true, "dtype": "float32", "function": "math.greater_equal"}}
?
?	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.__operators__.add_32", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.__operators__.add_32", "trainable": true, "dtype": "float32", "function": "__operators__.add"}}
?
?
embeddings
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Embedding", "name": "embedding_34", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "embedding_34", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 4, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 3]}}
?
?	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.cast_16", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.cast_16", "trainable": true, "dtype": "float32", "function": "cast"}}
?
?	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.__operators__.add_33", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.__operators__.add_33", "trainable": true, "dtype": "float32", "function": "__operators__.add"}}
?
?	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.expand_dims_11", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.expand_dims_11", "trainable": true, "dtype": "float32", "function": "expand_dims"}}
?
?	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.math.multiply_11", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.math.multiply_11", "trainable": true, "dtype": "float32", "function": "math.multiply"}}
?
?	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.math.reduce_sum_11", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.math.reduce_sum_11", "trainable": true, "dtype": "float32", "function": "math.reduce_sum"}}
8
?0
?1
?2"
trackable_list_wrapper
8
?0
?1
?2"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
Dtrainable_variables
 ?layer_regularization_losses
E	variables
Fregularization_losses
?layer_metrics
?metrics
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
#:!
??2dense_45/kernel
:?2dense_45/bias
.
L0
M1"
trackable_list_wrapper
.
L0
M1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
Ntrainable_variables
 ?layer_regularization_losses
O	variables
Pregularization_losses
?layer_metrics
?metrics
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
#:!
??2dense_46/kernel
:?2dense_46/bias
.
S0
T1"
trackable_list_wrapper
.
S0
T1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
Utrainable_variables
 ?layer_regularization_losses
V	variables
Wregularization_losses
?layer_metrics
?metrics
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
": 	?2dense_48/kernel
:?2dense_48/bias
.
Y0
Z1"
trackable_list_wrapper
.
Y0
Z1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
[trainable_variables
 ?layer_regularization_losses
\	variables
]regularization_losses
?layer_metrics
?metrics
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
#:!
??2dense_47/kernel
:?2dense_47/bias
.
_0
`1"
trackable_list_wrapper
.
_0
`1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
atrainable_variables
 ?layer_regularization_losses
b	variables
cregularization_losses
?layer_metrics
?metrics
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
#:!
??2dense_49/kernel
:?2dense_49/bias
.
e0
f1"
trackable_list_wrapper
.
e0
f1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
gtrainable_variables
 ?layer_regularization_losses
h	variables
iregularization_losses
?layer_metrics
?metrics
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
#:!
??2dense_50/kernel
:?2dense_50/bias
.
l0
m1"
trackable_list_wrapper
.
l0
m1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
ntrainable_variables
 ?layer_regularization_losses
o	variables
pregularization_losses
?layer_metrics
?metrics
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
#:!
??2dense_51/kernel
:?2dense_51/bias
.
s0
t1"
trackable_list_wrapper
.
s0
t1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
utrainable_variables
 ?layer_regularization_losses
v	variables
wregularization_losses
?layer_metrics
?metrics
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
#:!
??2dense_52/kernel
:?2dense_52/bias
.
{0
|1"
trackable_list_wrapper
.
{0
|1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
}trainable_variables
 ?layer_regularization_losses
~	variables
regularization_losses
?layer_metrics
?metrics
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
?
?state_variables
?_broadcast_shape
	?mean
?variance

?count
?	keras_api"?
_tf_keras_layer?{"class_name": "Normalization", "name": "normalization_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "normalization_5", "trainable": true, "dtype": "float32", "axis": {"class_name": "__tuple__", "items": [-1]}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 256]}}
 "
trackable_list_wrapper
8
?0
?1
?2"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
?trainable_variables
 ?layer_regularization_losses
?	variables
?regularization_losses
?layer_metrics
?metrics
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
": 	?2dense_53/kernel
:2dense_53/bias
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
?trainable_variables
 ?layer_regularization_losses
?	variables
?regularization_losses
?layer_metrics
?metrics
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
*:(	4?2embedding_32/embeddings
*:(	?2embedding_30/embeddings
*:(	?2embedding_31/embeddings
*:(	4?2embedding_35/embeddings
*:(	?2embedding_33/embeddings
*:(	?2embedding_34/embeddings
-:+?2 normalize_5/normalization_5/mean
1:/?2$normalize_5/normalization_5/variance
):'	 2!normalize_5/normalization_5/count
?
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
(
?0"
trackable_list_wrapper
8
?0
?1
?2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
?trainable_variables
 ?layer_regularization_losses
?	variables
?regularization_losses
?layer_metrics
?metrics
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
(
?0"
trackable_list_wrapper
(
?0"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
?trainable_variables
 ?layer_regularization_losses
?	variables
?regularization_losses
?layer_metrics
?metrics
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
(
?0"
trackable_list_wrapper
(
?0"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
?trainable_variables
 ?layer_regularization_losses
?	variables
?regularization_losses
?layer_metrics
?metrics
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
(
?0"
trackable_list_wrapper
(
?0"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
?trainable_variables
 ?layer_regularization_losses
?	variables
?regularization_losses
?layer_metrics
?metrics
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
?
"0
#1
$2
%3
&4
'5
(6
)7
*8
+9
,10
-11
.12
/13
014"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
?trainable_variables
 ?layer_regularization_losses
?	variables
?regularization_losses
?layer_metrics
?metrics
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
(
?0"
trackable_list_wrapper
(
?0"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
?trainable_variables
 ?layer_regularization_losses
?	variables
?regularization_losses
?layer_metrics
?metrics
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
(
?0"
trackable_list_wrapper
(
?0"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
?trainable_variables
 ?layer_regularization_losses
?	variables
?regularization_losses
?layer_metrics
?metrics
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
(
?0"
trackable_list_wrapper
(
?0"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
?trainable_variables
 ?layer_regularization_losses
?	variables
?regularization_losses
?layer_metrics
?metrics
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
?
50
61
72
83
94
:5
;6
<7
=8
>9
?10
@11
A12
B13
C14"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
F
	?mean
?variance

?count"
trackable_dict_wrapper
 "
trackable_list_wrapper
"
_generic_user_object
(
?0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
8
?0
?1
?2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?

?total

?count
?	variables
?	keras_api"?
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
(:&
??2Adam/dense_45/kernel/m
!:?2Adam/dense_45/bias/m
(:&
??2Adam/dense_46/kernel/m
!:?2Adam/dense_46/bias/m
':%	?2Adam/dense_48/kernel/m
!:?2Adam/dense_48/bias/m
(:&
??2Adam/dense_47/kernel/m
!:?2Adam/dense_47/bias/m
(:&
??2Adam/dense_49/kernel/m
!:?2Adam/dense_49/bias/m
(:&
??2Adam/dense_50/kernel/m
!:?2Adam/dense_50/bias/m
(:&
??2Adam/dense_51/kernel/m
!:?2Adam/dense_51/bias/m
(:&
??2Adam/dense_52/kernel/m
!:?2Adam/dense_52/bias/m
':%	?2Adam/dense_53/kernel/m
 :2Adam/dense_53/bias/m
/:-	4?2Adam/embedding_32/embeddings/m
/:-	?2Adam/embedding_30/embeddings/m
/:-	?2Adam/embedding_31/embeddings/m
/:-	4?2Adam/embedding_35/embeddings/m
/:-	?2Adam/embedding_33/embeddings/m
/:-	?2Adam/embedding_34/embeddings/m
(:&
??2Adam/dense_45/kernel/v
!:?2Adam/dense_45/bias/v
(:&
??2Adam/dense_46/kernel/v
!:?2Adam/dense_46/bias/v
':%	?2Adam/dense_48/kernel/v
!:?2Adam/dense_48/bias/v
(:&
??2Adam/dense_47/kernel/v
!:?2Adam/dense_47/bias/v
(:&
??2Adam/dense_49/kernel/v
!:?2Adam/dense_49/bias/v
(:&
??2Adam/dense_50/kernel/v
!:?2Adam/dense_50/bias/v
(:&
??2Adam/dense_51/kernel/v
!:?2Adam/dense_51/bias/v
(:&
??2Adam/dense_52/kernel/v
!:?2Adam/dense_52/bias/v
':%	?2Adam/dense_53/kernel/v
 :2Adam/dense_53/bias/v
/:-	4?2Adam/embedding_32/embeddings/v
/:-	?2Adam/embedding_30/embeddings/v
/:-	?2Adam/embedding_31/embeddings/v
/:-	4?2Adam/embedding_35/embeddings/v
/:-	?2Adam/embedding_33/embeddings/v
/:-	?2Adam/embedding_34/embeddings/v
?2?
#__inference__wrapped_model_41025355?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *q?n
l?i
G?D
 ?
cards0?????????
 ?
cards1?????????
?
bets?????????

?2?
1__inference_custom_model_5_layer_call_fn_41027075
1__inference_custom_model_5_layer_call_fn_41027144
1__inference_custom_model_5_layer_call_fn_41026587
1__inference_custom_model_5_layer_call_fn_41026426?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
L__inference_custom_model_5_layer_call_and_return_conditional_losses_41027006
L__inference_custom_model_5_layer_call_and_return_conditional_losses_41026836
L__inference_custom_model_5_layer_call_and_return_conditional_losses_41026264
L__inference_custom_model_5_layer_call_and_return_conditional_losses_41026172?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
+__inference_model_10_layer_call_fn_41027254
+__inference_model_10_layer_call_fn_41025536
+__inference_model_10_layer_call_fn_41027241
+__inference_model_10_layer_call_fn_41025581?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
F__inference_model_10_layer_call_and_return_conditional_losses_41027228
F__inference_model_10_layer_call_and_return_conditional_losses_41027186
F__inference_model_10_layer_call_and_return_conditional_losses_41025458
F__inference_model_10_layer_call_and_return_conditional_losses_41025490?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
+__inference_model_11_layer_call_fn_41025807
+__inference_model_11_layer_call_fn_41025762
+__inference_model_11_layer_call_fn_41027364
+__inference_model_11_layer_call_fn_41027351?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
F__inference_model_11_layer_call_and_return_conditional_losses_41027338
F__inference_model_11_layer_call_and_return_conditional_losses_41025684
F__inference_model_11_layer_call_and_return_conditional_losses_41025716
F__inference_model_11_layer_call_and_return_conditional_losses_41027296?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
+__inference_dense_45_layer_call_fn_41027384?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_dense_45_layer_call_and_return_conditional_losses_41027375?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
+__inference_dense_46_layer_call_fn_41027404?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_dense_46_layer_call_and_return_conditional_losses_41027395?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
+__inference_dense_48_layer_call_fn_41027423?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_dense_48_layer_call_and_return_conditional_losses_41027414?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
+__inference_dense_47_layer_call_fn_41027443?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_dense_47_layer_call_and_return_conditional_losses_41027434?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
+__inference_dense_49_layer_call_fn_41027462?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_dense_49_layer_call_and_return_conditional_losses_41027453?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
+__inference_dense_50_layer_call_fn_41027481?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_dense_50_layer_call_and_return_conditional_losses_41027472?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
+__inference_dense_51_layer_call_fn_41027500?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_dense_51_layer_call_and_return_conditional_losses_41027491?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
+__inference_dense_52_layer_call_fn_41027519?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_dense_52_layer_call_and_return_conditional_losses_41027510?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
.__inference_normalize_5_layer_call_fn_41027545?
???
FullArgSpec
args?
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
I__inference_normalize_5_layer_call_and_return_conditional_losses_41027536?
???
FullArgSpec
args?
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
+__inference_dense_53_layer_call_fn_41027564?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_dense_53_layer_call_and_return_conditional_losses_41027555?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
&__inference_signature_wrapper_41026666betscards0cards1"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
-__inference_flatten_10_layer_call_fn_41027575?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
H__inference_flatten_10_layer_call_and_return_conditional_losses_41027570?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
/__inference_embedding_32_layer_call_fn_41027592?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
J__inference_embedding_32_layer_call_and_return_conditional_losses_41027585?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
/__inference_embedding_30_layer_call_fn_41027609?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
J__inference_embedding_30_layer_call_and_return_conditional_losses_41027602?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
/__inference_embedding_31_layer_call_fn_41027626?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
J__inference_embedding_31_layer_call_and_return_conditional_losses_41027619?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
-__inference_flatten_11_layer_call_fn_41027637?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
H__inference_flatten_11_layer_call_and_return_conditional_losses_41027632?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
/__inference_embedding_35_layer_call_fn_41027654?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
J__inference_embedding_35_layer_call_and_return_conditional_losses_41027647?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
/__inference_embedding_33_layer_call_fn_41027671?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
J__inference_embedding_33_layer_call_and_return_conditional_losses_41027664?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
/__inference_embedding_34_layer_call_fn_41027688?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
J__inference_embedding_34_layer_call_and_return_conditional_losses_41027681?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
	J
Const
J	
Const_1
J	
Const_2
J	
Const_3
J	
Const_4?
#__inference__wrapped_model_41025355?.???????????LMYZST_`eflmst{|????{?x
q?n
l?i
G?D
 ?
cards0?????????
 ?
cards1?????????
?
bets?????????

? "3?0
.
dense_53"?
dense_53??????????
L__inference_custom_model_5_layer_call_and_return_conditional_losses_41026172?.???????????LMYZST_`eflmst{|???????
y?v
l?i
G?D
 ?
cards0?????????
 ?
cards1?????????
?
bets?????????

p

 
? "%?"
?
0?????????
? ?
L__inference_custom_model_5_layer_call_and_return_conditional_losses_41026264?.???????????LMYZST_`eflmst{|???????
y?v
l?i
G?D
 ?
cards0?????????
 ?
cards1?????????
?
bets?????????

p 

 
? "%?"
?
0?????????
? ?
L__inference_custom_model_5_layer_call_and_return_conditional_losses_41026836?.???????????LMYZST_`eflmst{|???????
???
x?u
O?L
$?!

inputs/0/0?????????
$?!

inputs/0/1?????????
"?
inputs/1?????????

p

 
? "%?"
?
0?????????
? ?
L__inference_custom_model_5_layer_call_and_return_conditional_losses_41027006?.???????????LMYZST_`eflmst{|???????
???
x?u
O?L
$?!

inputs/0/0?????????
$?!

inputs/0/1?????????
"?
inputs/1?????????

p 

 
? "%?"
?
0?????????
? ?
1__inference_custom_model_5_layer_call_fn_41026426?.???????????LMYZST_`eflmst{|???????
y?v
l?i
G?D
 ?
cards0?????????
 ?
cards1?????????
?
bets?????????

p

 
? "???????????
1__inference_custom_model_5_layer_call_fn_41026587?.???????????LMYZST_`eflmst{|???????
y?v
l?i
G?D
 ?
cards0?????????
 ?
cards1?????????
?
bets?????????

p 

 
? "???????????
1__inference_custom_model_5_layer_call_fn_41027075?.???????????LMYZST_`eflmst{|???????
???
x?u
O?L
$?!

inputs/0/0?????????
$?!

inputs/0/1?????????
"?
inputs/1?????????

p

 
? "???????????
1__inference_custom_model_5_layer_call_fn_41027144?.???????????LMYZST_`eflmst{|???????
???
x?u
O?L
$?!

inputs/0/0?????????
$?!

inputs/0/1?????????
"?
inputs/1?????????

p 

 
? "???????????
F__inference_dense_45_layer_call_and_return_conditional_losses_41027375^LM0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? ?
+__inference_dense_45_layer_call_fn_41027384QLM0?-
&?#
!?
inputs??????????
? "????????????
F__inference_dense_46_layer_call_and_return_conditional_losses_41027395^ST0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? ?
+__inference_dense_46_layer_call_fn_41027404QST0?-
&?#
!?
inputs??????????
? "????????????
F__inference_dense_47_layer_call_and_return_conditional_losses_41027434^_`0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? ?
+__inference_dense_47_layer_call_fn_41027443Q_`0?-
&?#
!?
inputs??????????
? "????????????
F__inference_dense_48_layer_call_and_return_conditional_losses_41027414]YZ/?,
%?"
 ?
inputs?????????
? "&?#
?
0??????????
? 
+__inference_dense_48_layer_call_fn_41027423PYZ/?,
%?"
 ?
inputs?????????
? "????????????
F__inference_dense_49_layer_call_and_return_conditional_losses_41027453^ef0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? ?
+__inference_dense_49_layer_call_fn_41027462Qef0?-
&?#
!?
inputs??????????
? "????????????
F__inference_dense_50_layer_call_and_return_conditional_losses_41027472^lm0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? ?
+__inference_dense_50_layer_call_fn_41027481Qlm0?-
&?#
!?
inputs??????????
? "????????????
F__inference_dense_51_layer_call_and_return_conditional_losses_41027491^st0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? ?
+__inference_dense_51_layer_call_fn_41027500Qst0?-
&?#
!?
inputs??????????
? "????????????
F__inference_dense_52_layer_call_and_return_conditional_losses_41027510^{|0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? ?
+__inference_dense_52_layer_call_fn_41027519Q{|0?-
&?#
!?
inputs??????????
? "????????????
F__inference_dense_53_layer_call_and_return_conditional_losses_41027555_??0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????
? ?
+__inference_dense_53_layer_call_fn_41027564R??0?-
&?#
!?
inputs??????????
? "???????????
J__inference_embedding_30_layer_call_and_return_conditional_losses_41027602a?/?,
%?"
 ?
inputs?????????
? "*?'
 ?
0??????????
? ?
/__inference_embedding_30_layer_call_fn_41027609T?/?,
%?"
 ?
inputs?????????
? "????????????
J__inference_embedding_31_layer_call_and_return_conditional_losses_41027619a?/?,
%?"
 ?
inputs?????????
? "*?'
 ?
0??????????
? ?
/__inference_embedding_31_layer_call_fn_41027626T?/?,
%?"
 ?
inputs?????????
? "????????????
J__inference_embedding_32_layer_call_and_return_conditional_losses_41027585a?/?,
%?"
 ?
inputs?????????
? "*?'
 ?
0??????????
? ?
/__inference_embedding_32_layer_call_fn_41027592T?/?,
%?"
 ?
inputs?????????
? "????????????
J__inference_embedding_33_layer_call_and_return_conditional_losses_41027664a?/?,
%?"
 ?
inputs?????????
? "*?'
 ?
0??????????
? ?
/__inference_embedding_33_layer_call_fn_41027671T?/?,
%?"
 ?
inputs?????????
? "????????????
J__inference_embedding_34_layer_call_and_return_conditional_losses_41027681a?/?,
%?"
 ?
inputs?????????
? "*?'
 ?
0??????????
? ?
/__inference_embedding_34_layer_call_fn_41027688T?/?,
%?"
 ?
inputs?????????
? "????????????
J__inference_embedding_35_layer_call_and_return_conditional_losses_41027647a?/?,
%?"
 ?
inputs?????????
? "*?'
 ?
0??????????
? ?
/__inference_embedding_35_layer_call_fn_41027654T?/?,
%?"
 ?
inputs?????????
? "????????????
H__inference_flatten_10_layer_call_and_return_conditional_losses_41027570X/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? |
-__inference_flatten_10_layer_call_fn_41027575K/?,
%?"
 ?
inputs?????????
? "???????????
H__inference_flatten_11_layer_call_and_return_conditional_losses_41027632X/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? |
-__inference_flatten_11_layer_call_fn_41027637K/?,
%?"
 ?
inputs?????????
? "???????????
F__inference_model_10_layer_call_and_return_conditional_losses_41025458m????9?6
/?,
"?
input_11?????????
p

 
? "&?#
?
0??????????
? ?
F__inference_model_10_layer_call_and_return_conditional_losses_41025490m????9?6
/?,
"?
input_11?????????
p 

 
? "&?#
?
0??????????
? ?
F__inference_model_10_layer_call_and_return_conditional_losses_41027186k????7?4
-?*
 ?
inputs?????????
p

 
? "&?#
?
0??????????
? ?
F__inference_model_10_layer_call_and_return_conditional_losses_41027228k????7?4
-?*
 ?
inputs?????????
p 

 
? "&?#
?
0??????????
? ?
+__inference_model_10_layer_call_fn_41025536`????9?6
/?,
"?
input_11?????????
p

 
? "????????????
+__inference_model_10_layer_call_fn_41025581`????9?6
/?,
"?
input_11?????????
p 

 
? "????????????
+__inference_model_10_layer_call_fn_41027241^????7?4
-?*
 ?
inputs?????????
p

 
? "????????????
+__inference_model_10_layer_call_fn_41027254^????7?4
-?*
 ?
inputs?????????
p 

 
? "????????????
F__inference_model_11_layer_call_and_return_conditional_losses_41025684m????9?6
/?,
"?
input_12?????????
p

 
? "&?#
?
0??????????
? ?
F__inference_model_11_layer_call_and_return_conditional_losses_41025716m????9?6
/?,
"?
input_12?????????
p 

 
? "&?#
?
0??????????
? ?
F__inference_model_11_layer_call_and_return_conditional_losses_41027296k????7?4
-?*
 ?
inputs?????????
p

 
? "&?#
?
0??????????
? ?
F__inference_model_11_layer_call_and_return_conditional_losses_41027338k????7?4
-?*
 ?
inputs?????????
p 

 
? "&?#
?
0??????????
? ?
+__inference_model_11_layer_call_fn_41025762`????9?6
/?,
"?
input_12?????????
p

 
? "????????????
+__inference_model_11_layer_call_fn_41025807`????9?6
/?,
"?
input_12?????????
p 

 
? "????????????
+__inference_model_11_layer_call_fn_41027351^????7?4
-?*
 ?
inputs?????????
p

 
? "????????????
+__inference_model_11_layer_call_fn_41027364^????7?4
-?*
 ?
inputs?????????
p 

 
? "????????????
I__inference_normalize_5_layer_call_and_return_conditional_losses_41027536[??+?(
!?
?
x??????????
? "&?#
?
0??????????
? ?
.__inference_normalize_5_layer_call_fn_41027545N??+?(
!?
?
x??????????
? "????????????
&__inference_signature_wrapper_41026666?.???????????LMYZST_`eflmst{|???????
? 
???
&
bets?
bets?????????

*
cards0 ?
cards0?????????
*
cards1 ?
cards1?????????"3?0
.
dense_53"?
dense_53?????????