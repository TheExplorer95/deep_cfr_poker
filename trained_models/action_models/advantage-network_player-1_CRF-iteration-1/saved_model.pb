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
dense_27/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??* 
shared_namedense_27/kernel
u
#dense_27/kernel/Read/ReadVariableOpReadVariableOpdense_27/kernel* 
_output_shapes
:
??*
dtype0
s
dense_27/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_27/bias
l
!dense_27/bias/Read/ReadVariableOpReadVariableOpdense_27/bias*
_output_shapes	
:?*
dtype0
|
dense_28/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??* 
shared_namedense_28/kernel
u
#dense_28/kernel/Read/ReadVariableOpReadVariableOpdense_28/kernel* 
_output_shapes
:
??*
dtype0
s
dense_28/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_28/bias
l
!dense_28/bias/Read/ReadVariableOpReadVariableOpdense_28/bias*
_output_shapes	
:?*
dtype0
{
dense_30/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?* 
shared_namedense_30/kernel
t
#dense_30/kernel/Read/ReadVariableOpReadVariableOpdense_30/kernel*
_output_shapes
:	?*
dtype0
s
dense_30/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_30/bias
l
!dense_30/bias/Read/ReadVariableOpReadVariableOpdense_30/bias*
_output_shapes	
:?*
dtype0
|
dense_29/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??* 
shared_namedense_29/kernel
u
#dense_29/kernel/Read/ReadVariableOpReadVariableOpdense_29/kernel* 
_output_shapes
:
??*
dtype0
s
dense_29/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_29/bias
l
!dense_29/bias/Read/ReadVariableOpReadVariableOpdense_29/bias*
_output_shapes	
:?*
dtype0
|
dense_31/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??* 
shared_namedense_31/kernel
u
#dense_31/kernel/Read/ReadVariableOpReadVariableOpdense_31/kernel* 
_output_shapes
:
??*
dtype0
s
dense_31/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_31/bias
l
!dense_31/bias/Read/ReadVariableOpReadVariableOpdense_31/bias*
_output_shapes	
:?*
dtype0
|
dense_32/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??* 
shared_namedense_32/kernel
u
#dense_32/kernel/Read/ReadVariableOpReadVariableOpdense_32/kernel* 
_output_shapes
:
??*
dtype0
s
dense_32/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_32/bias
l
!dense_32/bias/Read/ReadVariableOpReadVariableOpdense_32/bias*
_output_shapes	
:?*
dtype0
|
dense_33/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??* 
shared_namedense_33/kernel
u
#dense_33/kernel/Read/ReadVariableOpReadVariableOpdense_33/kernel* 
_output_shapes
:
??*
dtype0
s
dense_33/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_33/bias
l
!dense_33/bias/Read/ReadVariableOpReadVariableOpdense_33/bias*
_output_shapes	
:?*
dtype0
|
dense_34/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??* 
shared_namedense_34/kernel
u
#dense_34/kernel/Read/ReadVariableOpReadVariableOpdense_34/kernel* 
_output_shapes
:
??*
dtype0
s
dense_34/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_34/bias
l
!dense_34/bias/Read/ReadVariableOpReadVariableOpdense_34/bias*
_output_shapes	
:?*
dtype0
{
dense_35/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?* 
shared_namedense_35/kernel
t
#dense_35/kernel/Read/ReadVariableOpReadVariableOpdense_35/kernel*
_output_shapes
:	?*
dtype0
r
dense_35/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_35/bias
k
!dense_35/bias/Read/ReadVariableOpReadVariableOpdense_35/bias*
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
embedding_20/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	4?*(
shared_nameembedding_20/embeddings
?
+embedding_20/embeddings/Read/ReadVariableOpReadVariableOpembedding_20/embeddings*
_output_shapes
:	4?*
dtype0
?
embedding_18/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*(
shared_nameembedding_18/embeddings
?
+embedding_18/embeddings/Read/ReadVariableOpReadVariableOpembedding_18/embeddings*
_output_shapes
:	?*
dtype0
?
embedding_19/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*(
shared_nameembedding_19/embeddings
?
+embedding_19/embeddings/Read/ReadVariableOpReadVariableOpembedding_19/embeddings*
_output_shapes
:	?*
dtype0
?
embedding_23/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	4?*(
shared_nameembedding_23/embeddings
?
+embedding_23/embeddings/Read/ReadVariableOpReadVariableOpembedding_23/embeddings*
_output_shapes
:	4?*
dtype0
?
embedding_21/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*(
shared_nameembedding_21/embeddings
?
+embedding_21/embeddings/Read/ReadVariableOpReadVariableOpembedding_21/embeddings*
_output_shapes
:	?*
dtype0
?
embedding_22/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*(
shared_nameembedding_22/embeddings
?
+embedding_22/embeddings/Read/ReadVariableOpReadVariableOpembedding_22/embeddings*
_output_shapes
:	?*
dtype0
?
 normalize_3/normalization_3/meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*1
shared_name" normalize_3/normalization_3/mean
?
4normalize_3/normalization_3/mean/Read/ReadVariableOpReadVariableOp normalize_3/normalization_3/mean*
_output_shapes	
:?*
dtype0
?
$normalize_3/normalization_3/varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*5
shared_name&$normalize_3/normalization_3/variance
?
8normalize_3/normalization_3/variance/Read/ReadVariableOpReadVariableOp$normalize_3/normalization_3/variance*
_output_shapes	
:?*
dtype0
?
!normalize_3/normalization_3/countVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *2
shared_name#!normalize_3/normalization_3/count
?
5normalize_3/normalization_3/count/Read/ReadVariableOpReadVariableOp!normalize_3/normalization_3/count*
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
Adam/dense_27/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*'
shared_nameAdam/dense_27/kernel/m
?
*Adam/dense_27/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_27/kernel/m* 
_output_shapes
:
??*
dtype0
?
Adam/dense_27/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*%
shared_nameAdam/dense_27/bias/m
z
(Adam/dense_27/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_27/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/dense_28/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*'
shared_nameAdam/dense_28/kernel/m
?
*Adam/dense_28/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_28/kernel/m* 
_output_shapes
:
??*
dtype0
?
Adam/dense_28/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*%
shared_nameAdam/dense_28/bias/m
z
(Adam/dense_28/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_28/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/dense_30/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*'
shared_nameAdam/dense_30/kernel/m
?
*Adam/dense_30/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_30/kernel/m*
_output_shapes
:	?*
dtype0
?
Adam/dense_30/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*%
shared_nameAdam/dense_30/bias/m
z
(Adam/dense_30/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_30/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/dense_29/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*'
shared_nameAdam/dense_29/kernel/m
?
*Adam/dense_29/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_29/kernel/m* 
_output_shapes
:
??*
dtype0
?
Adam/dense_29/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*%
shared_nameAdam/dense_29/bias/m
z
(Adam/dense_29/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_29/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/dense_31/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*'
shared_nameAdam/dense_31/kernel/m
?
*Adam/dense_31/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_31/kernel/m* 
_output_shapes
:
??*
dtype0
?
Adam/dense_31/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*%
shared_nameAdam/dense_31/bias/m
z
(Adam/dense_31/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_31/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/dense_32/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*'
shared_nameAdam/dense_32/kernel/m
?
*Adam/dense_32/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_32/kernel/m* 
_output_shapes
:
??*
dtype0
?
Adam/dense_32/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*%
shared_nameAdam/dense_32/bias/m
z
(Adam/dense_32/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_32/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/dense_33/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*'
shared_nameAdam/dense_33/kernel/m
?
*Adam/dense_33/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_33/kernel/m* 
_output_shapes
:
??*
dtype0
?
Adam/dense_33/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*%
shared_nameAdam/dense_33/bias/m
z
(Adam/dense_33/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_33/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/dense_34/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*'
shared_nameAdam/dense_34/kernel/m
?
*Adam/dense_34/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_34/kernel/m* 
_output_shapes
:
??*
dtype0
?
Adam/dense_34/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*%
shared_nameAdam/dense_34/bias/m
z
(Adam/dense_34/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_34/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/dense_35/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*'
shared_nameAdam/dense_35/kernel/m
?
*Adam/dense_35/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_35/kernel/m*
_output_shapes
:	?*
dtype0
?
Adam/dense_35/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_35/bias/m
y
(Adam/dense_35/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_35/bias/m*
_output_shapes
:*
dtype0
?
Adam/embedding_20/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	4?*/
shared_name Adam/embedding_20/embeddings/m
?
2Adam/embedding_20/embeddings/m/Read/ReadVariableOpReadVariableOpAdam/embedding_20/embeddings/m*
_output_shapes
:	4?*
dtype0
?
Adam/embedding_18/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*/
shared_name Adam/embedding_18/embeddings/m
?
2Adam/embedding_18/embeddings/m/Read/ReadVariableOpReadVariableOpAdam/embedding_18/embeddings/m*
_output_shapes
:	?*
dtype0
?
Adam/embedding_19/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*/
shared_name Adam/embedding_19/embeddings/m
?
2Adam/embedding_19/embeddings/m/Read/ReadVariableOpReadVariableOpAdam/embedding_19/embeddings/m*
_output_shapes
:	?*
dtype0
?
Adam/embedding_23/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	4?*/
shared_name Adam/embedding_23/embeddings/m
?
2Adam/embedding_23/embeddings/m/Read/ReadVariableOpReadVariableOpAdam/embedding_23/embeddings/m*
_output_shapes
:	4?*
dtype0
?
Adam/embedding_21/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*/
shared_name Adam/embedding_21/embeddings/m
?
2Adam/embedding_21/embeddings/m/Read/ReadVariableOpReadVariableOpAdam/embedding_21/embeddings/m*
_output_shapes
:	?*
dtype0
?
Adam/embedding_22/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*/
shared_name Adam/embedding_22/embeddings/m
?
2Adam/embedding_22/embeddings/m/Read/ReadVariableOpReadVariableOpAdam/embedding_22/embeddings/m*
_output_shapes
:	?*
dtype0
?
Adam/dense_27/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*'
shared_nameAdam/dense_27/kernel/v
?
*Adam/dense_27/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_27/kernel/v* 
_output_shapes
:
??*
dtype0
?
Adam/dense_27/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*%
shared_nameAdam/dense_27/bias/v
z
(Adam/dense_27/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_27/bias/v*
_output_shapes	
:?*
dtype0
?
Adam/dense_28/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*'
shared_nameAdam/dense_28/kernel/v
?
*Adam/dense_28/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_28/kernel/v* 
_output_shapes
:
??*
dtype0
?
Adam/dense_28/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*%
shared_nameAdam/dense_28/bias/v
z
(Adam/dense_28/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_28/bias/v*
_output_shapes	
:?*
dtype0
?
Adam/dense_30/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*'
shared_nameAdam/dense_30/kernel/v
?
*Adam/dense_30/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_30/kernel/v*
_output_shapes
:	?*
dtype0
?
Adam/dense_30/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*%
shared_nameAdam/dense_30/bias/v
z
(Adam/dense_30/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_30/bias/v*
_output_shapes	
:?*
dtype0
?
Adam/dense_29/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*'
shared_nameAdam/dense_29/kernel/v
?
*Adam/dense_29/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_29/kernel/v* 
_output_shapes
:
??*
dtype0
?
Adam/dense_29/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*%
shared_nameAdam/dense_29/bias/v
z
(Adam/dense_29/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_29/bias/v*
_output_shapes	
:?*
dtype0
?
Adam/dense_31/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*'
shared_nameAdam/dense_31/kernel/v
?
*Adam/dense_31/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_31/kernel/v* 
_output_shapes
:
??*
dtype0
?
Adam/dense_31/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*%
shared_nameAdam/dense_31/bias/v
z
(Adam/dense_31/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_31/bias/v*
_output_shapes	
:?*
dtype0
?
Adam/dense_32/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*'
shared_nameAdam/dense_32/kernel/v
?
*Adam/dense_32/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_32/kernel/v* 
_output_shapes
:
??*
dtype0
?
Adam/dense_32/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*%
shared_nameAdam/dense_32/bias/v
z
(Adam/dense_32/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_32/bias/v*
_output_shapes	
:?*
dtype0
?
Adam/dense_33/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*'
shared_nameAdam/dense_33/kernel/v
?
*Adam/dense_33/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_33/kernel/v* 
_output_shapes
:
??*
dtype0
?
Adam/dense_33/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*%
shared_nameAdam/dense_33/bias/v
z
(Adam/dense_33/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_33/bias/v*
_output_shapes	
:?*
dtype0
?
Adam/dense_34/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*'
shared_nameAdam/dense_34/kernel/v
?
*Adam/dense_34/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_34/kernel/v* 
_output_shapes
:
??*
dtype0
?
Adam/dense_34/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*%
shared_nameAdam/dense_34/bias/v
z
(Adam/dense_34/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_34/bias/v*
_output_shapes	
:?*
dtype0
?
Adam/dense_35/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*'
shared_nameAdam/dense_35/kernel/v
?
*Adam/dense_35/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_35/kernel/v*
_output_shapes
:	?*
dtype0
?
Adam/dense_35/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_35/bias/v
y
(Adam/dense_35/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_35/bias/v*
_output_shapes
:*
dtype0
?
Adam/embedding_20/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	4?*/
shared_name Adam/embedding_20/embeddings/v
?
2Adam/embedding_20/embeddings/v/Read/ReadVariableOpReadVariableOpAdam/embedding_20/embeddings/v*
_output_shapes
:	4?*
dtype0
?
Adam/embedding_18/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*/
shared_name Adam/embedding_18/embeddings/v
?
2Adam/embedding_18/embeddings/v/Read/ReadVariableOpReadVariableOpAdam/embedding_18/embeddings/v*
_output_shapes
:	?*
dtype0
?
Adam/embedding_19/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*/
shared_name Adam/embedding_19/embeddings/v
?
2Adam/embedding_19/embeddings/v/Read/ReadVariableOpReadVariableOpAdam/embedding_19/embeddings/v*
_output_shapes
:	?*
dtype0
?
Adam/embedding_23/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	4?*/
shared_name Adam/embedding_23/embeddings/v
?
2Adam/embedding_23/embeddings/v/Read/ReadVariableOpReadVariableOpAdam/embedding_23/embeddings/v*
_output_shapes
:	4?*
dtype0
?
Adam/embedding_21/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*/
shared_name Adam/embedding_21/embeddings/v
?
2Adam/embedding_21/embeddings/v/Read/ReadVariableOpReadVariableOpAdam/embedding_21/embeddings/v*
_output_shapes
:	?*
dtype0
?
Adam/embedding_22/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*/
shared_name Adam/embedding_22/embeddings/v
?
2Adam/embedding_22/embeddings/v/Read/ReadVariableOpReadVariableOpAdam/embedding_22/embeddings/v*
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
regularization_losses
	variables
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
2regularization_losses
3	variables
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
Eregularization_losses
F	variables
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
Oregularization_losses
P	variables
Q	keras_api

R	keras_api
h

Skernel
Tbias
Utrainable_variables
Vregularization_losses
W	variables
X	keras_api
h

Ykernel
Zbias
[trainable_variables
\regularization_losses
]	variables
^	keras_api
h

_kernel
`bias
atrainable_variables
bregularization_losses
c	variables
d	keras_api
h

ekernel
fbias
gtrainable_variables
hregularization_losses
i	variables
j	keras_api

k	keras_api
h

lkernel
mbias
ntrainable_variables
oregularization_losses
p	variables
q	keras_api

r	keras_api
h

skernel
tbias
utrainable_variables
vregularization_losses
w	variables
x	keras_api

y	keras_api

z	keras_api
i

{kernel
|bias
}trainable_variables
~regularization_losses
	variables
?	keras_api

?	keras_api

?	keras_api
f
?	normalize
?trainable_variables
?regularization_losses
?	variables
?	keras_api
n
?kernel
	?bias
?trainable_variables
?regularization_losses
?	variables
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
?24
?25
?26
?
?layers
?metrics
 ?layer_regularization_losses
trainable_variables
?layer_metrics
regularization_losses
?non_trainable_variables
	variables
 
 
V
?trainable_variables
?regularization_losses
?	variables
?	keras_api

?	keras_api

?	keras_api
g
?
embeddings
?trainable_variables
?regularization_losses
?	variables
?	keras_api
g
?
embeddings
?trainable_variables
?regularization_losses
?	variables
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
?regularization_losses
?	variables
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
 

?0
?1
?2
?
?layers
?metrics
 ?layer_regularization_losses
1trainable_variables
?layer_metrics
2regularization_losses
?non_trainable_variables
3	variables
 
V
?trainable_variables
?regularization_losses
?	variables
?	keras_api

?	keras_api

?	keras_api
g
?
embeddings
?trainable_variables
?regularization_losses
?	variables
?	keras_api
g
?
embeddings
?trainable_variables
?regularization_losses
?	variables
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
?regularization_losses
?	variables
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
 

?0
?1
?2
?
?layers
?metrics
 ?layer_regularization_losses
Dtrainable_variables
?layer_metrics
Eregularization_losses
?non_trainable_variables
F	variables
 
 
 
 
[Y
VARIABLE_VALUEdense_27/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_27/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

L0
M1
 

L0
M1
?
?layers
?metrics
 ?layer_regularization_losses
Ntrainable_variables
?layer_metrics
Oregularization_losses
?non_trainable_variables
P	variables
 
[Y
VARIABLE_VALUEdense_28/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_28/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

S0
T1
 

S0
T1
?
?layers
?metrics
 ?layer_regularization_losses
Utrainable_variables
?layer_metrics
Vregularization_losses
?non_trainable_variables
W	variables
[Y
VARIABLE_VALUEdense_30/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_30/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

Y0
Z1
 

Y0
Z1
?
?layers
?metrics
 ?layer_regularization_losses
[trainable_variables
?layer_metrics
\regularization_losses
?non_trainable_variables
]	variables
[Y
VARIABLE_VALUEdense_29/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_29/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE

_0
`1
 

_0
`1
?
?layers
?metrics
 ?layer_regularization_losses
atrainable_variables
?layer_metrics
bregularization_losses
?non_trainable_variables
c	variables
[Y
VARIABLE_VALUEdense_31/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_31/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE

e0
f1
 

e0
f1
?
?layers
?metrics
 ?layer_regularization_losses
gtrainable_variables
?layer_metrics
hregularization_losses
?non_trainable_variables
i	variables
 
[Y
VARIABLE_VALUEdense_32/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_32/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE

l0
m1
 

l0
m1
?
?layers
?metrics
 ?layer_regularization_losses
ntrainable_variables
?layer_metrics
oregularization_losses
?non_trainable_variables
p	variables
 
[Y
VARIABLE_VALUEdense_33/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_33/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE

s0
t1
 

s0
t1
?
?layers
?metrics
 ?layer_regularization_losses
utrainable_variables
?layer_metrics
vregularization_losses
?non_trainable_variables
w	variables
 
 
[Y
VARIABLE_VALUEdense_34/kernel6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_34/bias4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUE

{0
|1
 

{0
|1
?
?layers
?metrics
 ?layer_regularization_losses
}trainable_variables
?layer_metrics
~regularization_losses
?non_trainable_variables
	variables
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
 

?0
?1
?2
?
?layers
?metrics
 ?layer_regularization_losses
?trainable_variables
?layer_metrics
?regularization_losses
?non_trainable_variables
?	variables
\Z
VARIABLE_VALUEdense_35/kernel7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdense_35/bias5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUE

?0
?1
 

?0
?1
?
?layers
?metrics
 ?layer_regularization_losses
?trainable_variables
?layer_metrics
?regularization_losses
?non_trainable_variables
?	variables
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
VARIABLE_VALUEembedding_20/embeddings0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUEembedding_18/embeddings0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUEembedding_19/embeddings0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUEembedding_23/embeddings0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUEembedding_21/embeddings0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUEembedding_22/embeddings0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUE normalize_3/normalization_3/mean'variables/22/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUE$normalize_3/normalization_3/variance'variables/23/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUE!normalize_3/normalization_3/count'variables/24/.ATTRIBUTES/VARIABLE_VALUE
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

?0
 
 

?0
?1
?2
 
 
 
?
?layers
?metrics
 ?layer_regularization_losses
?trainable_variables
?layer_metrics
?regularization_losses
?non_trainable_variables
?	variables
 
 

?0
 

?0
?
?layers
?metrics
 ?layer_regularization_losses
?trainable_variables
?layer_metrics
?regularization_losses
?non_trainable_variables
?	variables

?0
 

?0
?
?layers
?metrics
 ?layer_regularization_losses
?trainable_variables
?layer_metrics
?regularization_losses
?non_trainable_variables
?	variables
 
 
 

?0
 

?0
?
?layers
?metrics
 ?layer_regularization_losses
?trainable_variables
?layer_metrics
?regularization_losses
?non_trainable_variables
?	variables
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
?metrics
 ?layer_regularization_losses
?trainable_variables
?layer_metrics
?regularization_losses
?non_trainable_variables
?	variables
 
 

?0
 

?0
?
?layers
?metrics
 ?layer_regularization_losses
?trainable_variables
?layer_metrics
?regularization_losses
?non_trainable_variables
?	variables

?0
 

?0
?
?layers
?metrics
 ?layer_regularization_losses
?trainable_variables
?layer_metrics
?regularization_losses
?non_trainable_variables
?	variables
 
 
 

?0
 

?0
?
?layers
?metrics
 ?layer_regularization_losses
?trainable_variables
?layer_metrics
?regularization_losses
?non_trainable_variables
?	variables
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
VARIABLE_VALUEAdam/dense_27/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_27/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_28/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_28/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_30/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_30/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_29/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_29/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_31/kernel/mRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_31/bias/mPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_32/kernel/mRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_32/bias/mPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_33/kernel/mRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_33/bias/mPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_34/kernel/mRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_34/bias/mPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_35/kernel/mSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_35/bias/mQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/embedding_20/embeddings/mLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/embedding_18/embeddings/mLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/embedding_19/embeddings/mLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/embedding_23/embeddings/mLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/embedding_21/embeddings/mLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/embedding_22/embeddings/mLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_27/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_27/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_28/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_28/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_30/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_30/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_29/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_29/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_31/kernel/vRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_31/bias/vPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_32/kernel/vRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_32/bias/vPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_33/kernel/vRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_33/bias/vPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_34/kernel/vRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_34/bias/vPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_35/kernel/vSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_35/bias/vQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/embedding_20/embeddings/vLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/embedding_18/embeddings/vLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/embedding_19/embeddings/vLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/embedding_23/embeddings/vLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/embedding_21/embeddings/vLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/embedding_22/embeddings/vLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
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
StatefulPartitionedCallStatefulPartitionedCallserving_default_betsserving_default_cards0serving_default_cards1ConstConst_1embedding_20/embeddingsembedding_18/embeddingsembedding_19/embeddingsConst_2embedding_23/embeddingsembedding_21/embeddingsembedding_22/embeddingsConst_3Const_4dense_27/kerneldense_27/biasdense_30/kerneldense_30/biasdense_28/kerneldense_28/biasdense_29/kerneldense_29/biasdense_31/kerneldense_31/biasdense_32/kerneldense_32/biasdense_33/kerneldense_33/biasdense_34/kerneldense_34/bias normalize_3/normalization_3/mean$normalize_3/normalization_3/variancedense_35/kerneldense_35/bias*-
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
&__inference_signature_wrapper_20518682
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#dense_27/kernel/Read/ReadVariableOp!dense_27/bias/Read/ReadVariableOp#dense_28/kernel/Read/ReadVariableOp!dense_28/bias/Read/ReadVariableOp#dense_30/kernel/Read/ReadVariableOp!dense_30/bias/Read/ReadVariableOp#dense_29/kernel/Read/ReadVariableOp!dense_29/bias/Read/ReadVariableOp#dense_31/kernel/Read/ReadVariableOp!dense_31/bias/Read/ReadVariableOp#dense_32/kernel/Read/ReadVariableOp!dense_32/bias/Read/ReadVariableOp#dense_33/kernel/Read/ReadVariableOp!dense_33/bias/Read/ReadVariableOp#dense_34/kernel/Read/ReadVariableOp!dense_34/bias/Read/ReadVariableOp#dense_35/kernel/Read/ReadVariableOp!dense_35/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp+embedding_20/embeddings/Read/ReadVariableOp+embedding_18/embeddings/Read/ReadVariableOp+embedding_19/embeddings/Read/ReadVariableOp+embedding_23/embeddings/Read/ReadVariableOp+embedding_21/embeddings/Read/ReadVariableOp+embedding_22/embeddings/Read/ReadVariableOp4normalize_3/normalization_3/mean/Read/ReadVariableOp8normalize_3/normalization_3/variance/Read/ReadVariableOp5normalize_3/normalization_3/count/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp*Adam/dense_27/kernel/m/Read/ReadVariableOp(Adam/dense_27/bias/m/Read/ReadVariableOp*Adam/dense_28/kernel/m/Read/ReadVariableOp(Adam/dense_28/bias/m/Read/ReadVariableOp*Adam/dense_30/kernel/m/Read/ReadVariableOp(Adam/dense_30/bias/m/Read/ReadVariableOp*Adam/dense_29/kernel/m/Read/ReadVariableOp(Adam/dense_29/bias/m/Read/ReadVariableOp*Adam/dense_31/kernel/m/Read/ReadVariableOp(Adam/dense_31/bias/m/Read/ReadVariableOp*Adam/dense_32/kernel/m/Read/ReadVariableOp(Adam/dense_32/bias/m/Read/ReadVariableOp*Adam/dense_33/kernel/m/Read/ReadVariableOp(Adam/dense_33/bias/m/Read/ReadVariableOp*Adam/dense_34/kernel/m/Read/ReadVariableOp(Adam/dense_34/bias/m/Read/ReadVariableOp*Adam/dense_35/kernel/m/Read/ReadVariableOp(Adam/dense_35/bias/m/Read/ReadVariableOp2Adam/embedding_20/embeddings/m/Read/ReadVariableOp2Adam/embedding_18/embeddings/m/Read/ReadVariableOp2Adam/embedding_19/embeddings/m/Read/ReadVariableOp2Adam/embedding_23/embeddings/m/Read/ReadVariableOp2Adam/embedding_21/embeddings/m/Read/ReadVariableOp2Adam/embedding_22/embeddings/m/Read/ReadVariableOp*Adam/dense_27/kernel/v/Read/ReadVariableOp(Adam/dense_27/bias/v/Read/ReadVariableOp*Adam/dense_28/kernel/v/Read/ReadVariableOp(Adam/dense_28/bias/v/Read/ReadVariableOp*Adam/dense_30/kernel/v/Read/ReadVariableOp(Adam/dense_30/bias/v/Read/ReadVariableOp*Adam/dense_29/kernel/v/Read/ReadVariableOp(Adam/dense_29/bias/v/Read/ReadVariableOp*Adam/dense_31/kernel/v/Read/ReadVariableOp(Adam/dense_31/bias/v/Read/ReadVariableOp*Adam/dense_32/kernel/v/Read/ReadVariableOp(Adam/dense_32/bias/v/Read/ReadVariableOp*Adam/dense_33/kernel/v/Read/ReadVariableOp(Adam/dense_33/bias/v/Read/ReadVariableOp*Adam/dense_34/kernel/v/Read/ReadVariableOp(Adam/dense_34/bias/v/Read/ReadVariableOp*Adam/dense_35/kernel/v/Read/ReadVariableOp(Adam/dense_35/bias/v/Read/ReadVariableOp2Adam/embedding_20/embeddings/v/Read/ReadVariableOp2Adam/embedding_18/embeddings/v/Read/ReadVariableOp2Adam/embedding_19/embeddings/v/Read/ReadVariableOp2Adam/embedding_23/embeddings/v/Read/ReadVariableOp2Adam/embedding_21/embeddings/v/Read/ReadVariableOp2Adam/embedding_22/embeddings/v/Read/ReadVariableOpConst_5*_
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
!__inference__traced_save_20519980
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_27/kerneldense_27/biasdense_28/kerneldense_28/biasdense_30/kerneldense_30/biasdense_29/kerneldense_29/biasdense_31/kerneldense_31/biasdense_32/kerneldense_32/biasdense_33/kerneldense_33/biasdense_34/kerneldense_34/biasdense_35/kerneldense_35/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_rateembedding_20/embeddingsembedding_18/embeddingsembedding_19/embeddingsembedding_23/embeddingsembedding_21/embeddingsembedding_22/embeddings normalize_3/normalization_3/mean$normalize_3/normalization_3/variance!normalize_3/normalization_3/counttotalcountAdam/dense_27/kernel/mAdam/dense_27/bias/mAdam/dense_28/kernel/mAdam/dense_28/bias/mAdam/dense_30/kernel/mAdam/dense_30/bias/mAdam/dense_29/kernel/mAdam/dense_29/bias/mAdam/dense_31/kernel/mAdam/dense_31/bias/mAdam/dense_32/kernel/mAdam/dense_32/bias/mAdam/dense_33/kernel/mAdam/dense_33/bias/mAdam/dense_34/kernel/mAdam/dense_34/bias/mAdam/dense_35/kernel/mAdam/dense_35/bias/mAdam/embedding_20/embeddings/mAdam/embedding_18/embeddings/mAdam/embedding_19/embeddings/mAdam/embedding_23/embeddings/mAdam/embedding_21/embeddings/mAdam/embedding_22/embeddings/mAdam/dense_27/kernel/vAdam/dense_27/bias/vAdam/dense_28/kernel/vAdam/dense_28/bias/vAdam/dense_30/kernel/vAdam/dense_30/bias/vAdam/dense_29/kernel/vAdam/dense_29/bias/vAdam/dense_31/kernel/vAdam/dense_31/bias/vAdam/dense_32/kernel/vAdam/dense_32/bias/vAdam/dense_33/kernel/vAdam/dense_33/bias/vAdam/dense_34/kernel/vAdam/dense_34/bias/vAdam/dense_35/kernel/vAdam/dense_35/bias/vAdam/embedding_20/embeddings/vAdam/embedding_18/embeddings/vAdam/embedding_19/embeddings/vAdam/embedding_23/embeddings/vAdam/embedding_21/embeddings/vAdam/embedding_22/embeddings/v*^
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
$__inference__traced_restore_20520236??
?	
?
F__inference_dense_33_layer_call_and_return_conditional_losses_20519507

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
J__inference_embedding_21_layer_call_and_return_conditional_losses_20519680

inputs
embedding_lookup_20519674
identity??embedding_lookup]
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:?????????2
Cast?
embedding_lookupResourceGatherembedding_lookup_20519674Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*,
_class"
 loc:@embedding_lookup/20519674*,
_output_shapes
:??????????*
dtype02
embedding_lookup?
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*,
_class"
 loc:@embedding_lookup/20519674*,
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

L__inference_custom_model_3_layer_call_and_return_conditional_losses_20518538

inputs
inputs_1
inputs_2+
'tf_math_greater_equal_11_greaterequal_y
model_6_20518453
model_6_20518455
model_6_20518457
model_6_20518459
model_7_20518462
model_7_20518464
model_7_20518466
model_7_20518468/
+tf_clip_by_value_11_clip_by_value_minimum_y'
#tf_clip_by_value_11_clip_by_value_y
dense_27_20518480
dense_27_20518482
dense_30_20518485
dense_30_20518487
dense_28_20518490
dense_28_20518492
dense_29_20518495
dense_29_20518497
dense_31_20518500
dense_31_20518502
dense_32_20518507
dense_32_20518509
dense_33_20518513
dense_33_20518515
dense_34_20518520
dense_34_20518522
normalize_3_20518527
normalize_3_20518529
dense_35_20518532
dense_35_20518534
identity?? dense_27/StatefulPartitionedCall? dense_28/StatefulPartitionedCall? dense_29/StatefulPartitionedCall? dense_30/StatefulPartitionedCall? dense_31/StatefulPartitionedCall? dense_32/StatefulPartitionedCall? dense_33/StatefulPartitionedCall? dense_34/StatefulPartitionedCall? dense_35/StatefulPartitionedCall?model_6/StatefulPartitionedCall?model_7/StatefulPartitionedCall?#normalize_3/StatefulPartitionedCall?
%tf.math.greater_equal_11/GreaterEqualGreaterEqualinputs_2'tf_math_greater_equal_11_greaterequal_y*
T0*'
_output_shapes
:?????????
2'
%tf.math.greater_equal_11/GreaterEqual?
model_6/StatefulPartitionedCallStatefulPartitionedCallinputsmodel_6_20518453model_6_20518455model_6_20518457model_6_20518459*
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
GPU2 *0J 8? *N
fIRG
E__inference_model_6_layer_call_and_return_conditional_losses_205175862!
model_6/StatefulPartitionedCall?
model_7/StatefulPartitionedCallStatefulPartitionedCallinputs_1model_7_20518462model_7_20518464model_7_20518466model_7_20518468*
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
GPU2 *0J 8? *N
fIRG
E__inference_model_7_layer_call_and_return_conditional_losses_205178122!
model_7/StatefulPartitionedCall?
)tf.clip_by_value_11/clip_by_value/MinimumMinimuminputs_2+tf_clip_by_value_11_clip_by_value_minimum_y*
T0*'
_output_shapes
:?????????
2+
)tf.clip_by_value_11/clip_by_value/Minimum?
!tf.clip_by_value_11/clip_by_valueMaximum-tf.clip_by_value_11/clip_by_value/Minimum:z:0#tf_clip_by_value_11_clip_by_value_y*
T0*'
_output_shapes
:?????????
2#
!tf.clip_by_value_11/clip_by_value?
tf.cast_11/CastCast)tf.math.greater_equal_11/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????
2
tf.cast_11/Castt
tf.concat_9/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
tf.concat_9/concat/axis?
tf.concat_9/concatConcatV2(model_6/StatefulPartitionedCall:output:0(model_7/StatefulPartitionedCall:output:0 tf.concat_9/concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????2
tf.concat_9/concat
tf.concat_10/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
tf.concat_10/concat/axis?
tf.concat_10/concatConcatV2%tf.clip_by_value_11/clip_by_value:z:0tf.cast_11/Cast:y:0!tf.concat_10/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????2
tf.concat_10/concat?
 dense_27/StatefulPartitionedCallStatefulPartitionedCalltf.concat_9/concat:output:0dense_27_20518480dense_27_20518482*
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
F__inference_dense_27_layer_call_and_return_conditional_losses_205179212"
 dense_27/StatefulPartitionedCall?
 dense_30/StatefulPartitionedCallStatefulPartitionedCalltf.concat_10/concat:output:0dense_30_20518485dense_30_20518487*
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
F__inference_dense_30_layer_call_and_return_conditional_losses_205179472"
 dense_30/StatefulPartitionedCall?
 dense_28/StatefulPartitionedCallStatefulPartitionedCall)dense_27/StatefulPartitionedCall:output:0dense_28_20518490dense_28_20518492*
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
F__inference_dense_28_layer_call_and_return_conditional_losses_205179742"
 dense_28/StatefulPartitionedCall?
 dense_29/StatefulPartitionedCallStatefulPartitionedCall)dense_28/StatefulPartitionedCall:output:0dense_29_20518495dense_29_20518497*
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
F__inference_dense_29_layer_call_and_return_conditional_losses_205180012"
 dense_29/StatefulPartitionedCall?
 dense_31/StatefulPartitionedCallStatefulPartitionedCall)dense_30/StatefulPartitionedCall:output:0dense_31_20518500dense_31_20518502*
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
F__inference_dense_31_layer_call_and_return_conditional_losses_205180272"
 dense_31/StatefulPartitionedCall
tf.concat_11/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
tf.concat_11/concat/axis?
tf.concat_11/concatConcatV2)dense_29/StatefulPartitionedCall:output:0)dense_31/StatefulPartitionedCall:output:0!tf.concat_11/concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????2
tf.concat_11/concat?
 dense_32/StatefulPartitionedCallStatefulPartitionedCalltf.concat_11/concat:output:0dense_32_20518507dense_32_20518509*
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
F__inference_dense_32_layer_call_and_return_conditional_losses_205180552"
 dense_32/StatefulPartitionedCall?
tf.nn.relu_9/ReluRelu)dense_32/StatefulPartitionedCall:output:0*
T0*(
_output_shapes
:??????????2
tf.nn.relu_9/Relu?
 dense_33/StatefulPartitionedCallStatefulPartitionedCalltf.nn.relu_9/Relu:activations:0dense_33_20518513dense_33_20518515*
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
F__inference_dense_33_layer_call_and_return_conditional_losses_205180822"
 dense_33/StatefulPartitionedCall?
tf.__operators__.add_22/AddV2AddV2)dense_33/StatefulPartitionedCall:output:0tf.nn.relu_9/Relu:activations:0*
T0*(
_output_shapes
:??????????2
tf.__operators__.add_22/AddV2?
tf.nn.relu_10/ReluRelu!tf.__operators__.add_22/AddV2:z:0*
T0*(
_output_shapes
:??????????2
tf.nn.relu_10/Relu?
 dense_34/StatefulPartitionedCallStatefulPartitionedCall tf.nn.relu_10/Relu:activations:0dense_34_20518520dense_34_20518522*
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
F__inference_dense_34_layer_call_and_return_conditional_losses_205181102"
 dense_34/StatefulPartitionedCall?
tf.__operators__.add_23/AddV2AddV2)dense_34/StatefulPartitionedCall:output:0 tf.nn.relu_10/Relu:activations:0*
T0*(
_output_shapes
:??????????2
tf.__operators__.add_23/AddV2?
tf.nn.relu_11/ReluRelu!tf.__operators__.add_23/AddV2:z:0*
T0*(
_output_shapes
:??????????2
tf.nn.relu_11/Relu?
#normalize_3/StatefulPartitionedCallStatefulPartitionedCall tf.nn.relu_11/Relu:activations:0normalize_3_20518527normalize_3_20518529*
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
I__inference_normalize_3_layer_call_and_return_conditional_losses_205181452%
#normalize_3/StatefulPartitionedCall?
 dense_35/StatefulPartitionedCallStatefulPartitionedCall,normalize_3/StatefulPartitionedCall:output:0dense_35_20518532dense_35_20518534*
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
F__inference_dense_35_layer_call_and_return_conditional_losses_205181712"
 dense_35/StatefulPartitionedCall?
IdentityIdentity)dense_35/StatefulPartitionedCall:output:0!^dense_27/StatefulPartitionedCall!^dense_28/StatefulPartitionedCall!^dense_29/StatefulPartitionedCall!^dense_30/StatefulPartitionedCall!^dense_31/StatefulPartitionedCall!^dense_32/StatefulPartitionedCall!^dense_33/StatefulPartitionedCall!^dense_34/StatefulPartitionedCall!^dense_35/StatefulPartitionedCall ^model_6/StatefulPartitionedCall ^model_7/StatefulPartitionedCall$^normalize_3/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????:?????????:?????????
: : :::: :::: : ::::::::::::::::::::2D
 dense_27/StatefulPartitionedCall dense_27/StatefulPartitionedCall2D
 dense_28/StatefulPartitionedCall dense_28/StatefulPartitionedCall2D
 dense_29/StatefulPartitionedCall dense_29/StatefulPartitionedCall2D
 dense_30/StatefulPartitionedCall dense_30/StatefulPartitionedCall2D
 dense_31/StatefulPartitionedCall dense_31/StatefulPartitionedCall2D
 dense_32/StatefulPartitionedCall dense_32/StatefulPartitionedCall2D
 dense_33/StatefulPartitionedCall dense_33/StatefulPartitionedCall2D
 dense_34/StatefulPartitionedCall dense_34/StatefulPartitionedCall2D
 dense_35/StatefulPartitionedCall dense_35/StatefulPartitionedCall2B
model_6/StatefulPartitionedCallmodel_6/StatefulPartitionedCall2B
model_7/StatefulPartitionedCallmodel_7/StatefulPartitionedCall2J
#normalize_3/StatefulPartitionedCall#normalize_3/StatefulPartitionedCall:O K
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
??
?
L__inference_custom_model_3_layer_call_and_return_conditional_losses_20518852

inputs_0_0

inputs_0_1
inputs_1+
'tf_math_greater_equal_11_greaterequal_y2
.model_6_tf_math_greater_equal_9_greaterequal_y2
.model_6_embedding_20_embedding_lookup_205187022
.model_6_embedding_18_embedding_lookup_205187082
.model_6_embedding_19_embedding_lookup_205187163
/model_7_tf_math_greater_equal_10_greaterequal_y2
.model_7_embedding_23_embedding_lookup_205187402
.model_7_embedding_21_embedding_lookup_205187462
.model_7_embedding_22_embedding_lookup_20518754/
+tf_clip_by_value_11_clip_by_value_minimum_y'
#tf_clip_by_value_11_clip_by_value_y+
'dense_27_matmul_readvariableop_resource,
(dense_27_biasadd_readvariableop_resource+
'dense_30_matmul_readvariableop_resource,
(dense_30_biasadd_readvariableop_resource+
'dense_28_matmul_readvariableop_resource,
(dense_28_biasadd_readvariableop_resource+
'dense_29_matmul_readvariableop_resource,
(dense_29_biasadd_readvariableop_resource+
'dense_31_matmul_readvariableop_resource,
(dense_31_biasadd_readvariableop_resource+
'dense_32_matmul_readvariableop_resource,
(dense_32_biasadd_readvariableop_resource+
'dense_33_matmul_readvariableop_resource,
(dense_33_biasadd_readvariableop_resource+
'dense_34_matmul_readvariableop_resource,
(dense_34_biasadd_readvariableop_resource?
;normalize_3_normalization_3_reshape_readvariableop_resourceA
=normalize_3_normalization_3_reshape_1_readvariableop_resource+
'dense_35_matmul_readvariableop_resource,
(dense_35_biasadd_readvariableop_resource
identity??dense_27/BiasAdd/ReadVariableOp?dense_27/MatMul/ReadVariableOp?dense_28/BiasAdd/ReadVariableOp?dense_28/MatMul/ReadVariableOp?dense_29/BiasAdd/ReadVariableOp?dense_29/MatMul/ReadVariableOp?dense_30/BiasAdd/ReadVariableOp?dense_30/MatMul/ReadVariableOp?dense_31/BiasAdd/ReadVariableOp?dense_31/MatMul/ReadVariableOp?dense_32/BiasAdd/ReadVariableOp?dense_32/MatMul/ReadVariableOp?dense_33/BiasAdd/ReadVariableOp?dense_33/MatMul/ReadVariableOp?dense_34/BiasAdd/ReadVariableOp?dense_34/MatMul/ReadVariableOp?dense_35/BiasAdd/ReadVariableOp?dense_35/MatMul/ReadVariableOp?%model_6/embedding_18/embedding_lookup?%model_6/embedding_19/embedding_lookup?%model_6/embedding_20/embedding_lookup?%model_7/embedding_21/embedding_lookup?%model_7/embedding_22/embedding_lookup?%model_7/embedding_23/embedding_lookup?2normalize_3/normalization_3/Reshape/ReadVariableOp?4normalize_3/normalization_3/Reshape_1/ReadVariableOp?
%tf.math.greater_equal_11/GreaterEqualGreaterEqualinputs_1'tf_math_greater_equal_11_greaterequal_y*
T0*'
_output_shapes
:?????????
2'
%tf.math.greater_equal_11/GreaterEqual?
model_6/flatten_6/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
model_6/flatten_6/Const?
model_6/flatten_6/ReshapeReshape
inputs_0_0 model_6/flatten_6/Const:output:0*
T0*'
_output_shapes
:?????????2
model_6/flatten_6/Reshape?
2model_6/tf.clip_by_value_9/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 * $tI24
2model_6/tf.clip_by_value_9/clip_by_value/Minimum/y?
0model_6/tf.clip_by_value_9/clip_by_value/MinimumMinimum"model_6/flatten_6/Reshape:output:0;model_6/tf.clip_by_value_9/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????22
0model_6/tf.clip_by_value_9/clip_by_value/Minimum?
*model_6/tf.clip_by_value_9/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2,
*model_6/tf.clip_by_value_9/clip_by_value/y?
(model_6/tf.clip_by_value_9/clip_by_valueMaximum4model_6/tf.clip_by_value_9/clip_by_value/Minimum:z:03model_6/tf.clip_by_value_9/clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????2*
(model_6/tf.clip_by_value_9/clip_by_value?
+model_6/tf.compat.v1.floor_div_6/FloorDiv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2-
+model_6/tf.compat.v1.floor_div_6/FloorDiv/y?
)model_6/tf.compat.v1.floor_div_6/FloorDivFloorDiv,model_6/tf.clip_by_value_9/clip_by_value:z:04model_6/tf.compat.v1.floor_div_6/FloorDiv/y:output:0*
T0*'
_output_shapes
:?????????2+
)model_6/tf.compat.v1.floor_div_6/FloorDiv?
,model_6/tf.math.greater_equal_9/GreaterEqualGreaterEqual"model_6/flatten_6/Reshape:output:0.model_6_tf_math_greater_equal_9_greaterequal_y*
T0*'
_output_shapes
:?????????2.
,model_6/tf.math.greater_equal_9/GreaterEqual?
%model_6/tf.math.floormod_6/FloorMod/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2'
%model_6/tf.math.floormod_6/FloorMod/y?
#model_6/tf.math.floormod_6/FloorModFloorMod,model_6/tf.clip_by_value_9/clip_by_value:z:0.model_6/tf.math.floormod_6/FloorMod/y:output:0*
T0*'
_output_shapes
:?????????2%
#model_6/tf.math.floormod_6/FloorMod?
model_6/embedding_20/CastCast,model_6/tf.clip_by_value_9/clip_by_value:z:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2
model_6/embedding_20/Cast?
%model_6/embedding_20/embedding_lookupResourceGather.model_6_embedding_20_embedding_lookup_20518702model_6/embedding_20/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*A
_class7
53loc:@model_6/embedding_20/embedding_lookup/20518702*,
_output_shapes
:??????????*
dtype02'
%model_6/embedding_20/embedding_lookup?
.model_6/embedding_20/embedding_lookup/IdentityIdentity.model_6/embedding_20/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*A
_class7
53loc:@model_6/embedding_20/embedding_lookup/20518702*,
_output_shapes
:??????????20
.model_6/embedding_20/embedding_lookup/Identity?
0model_6/embedding_20/embedding_lookup/Identity_1Identity7model_6/embedding_20/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????22
0model_6/embedding_20/embedding_lookup/Identity_1?
model_6/embedding_18/CastCast-model_6/tf.compat.v1.floor_div_6/FloorDiv:z:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2
model_6/embedding_18/Cast?
%model_6/embedding_18/embedding_lookupResourceGather.model_6_embedding_18_embedding_lookup_20518708model_6/embedding_18/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*A
_class7
53loc:@model_6/embedding_18/embedding_lookup/20518708*,
_output_shapes
:??????????*
dtype02'
%model_6/embedding_18/embedding_lookup?
.model_6/embedding_18/embedding_lookup/IdentityIdentity.model_6/embedding_18/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*A
_class7
53loc:@model_6/embedding_18/embedding_lookup/20518708*,
_output_shapes
:??????????20
.model_6/embedding_18/embedding_lookup/Identity?
0model_6/embedding_18/embedding_lookup/Identity_1Identity7model_6/embedding_18/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????22
0model_6/embedding_18/embedding_lookup/Identity_1?
model_6/tf.cast_9/CastCast0model_6/tf.math.greater_equal_9/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2
model_6/tf.cast_9/Cast?
%model_6/tf.__operators__.add_18/AddV2AddV29model_6/embedding_20/embedding_lookup/Identity_1:output:09model_6/embedding_18/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:??????????2'
%model_6/tf.__operators__.add_18/AddV2?
model_6/embedding_19/CastCast'model_6/tf.math.floormod_6/FloorMod:z:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2
model_6/embedding_19/Cast?
%model_6/embedding_19/embedding_lookupResourceGather.model_6_embedding_19_embedding_lookup_20518716model_6/embedding_19/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*A
_class7
53loc:@model_6/embedding_19/embedding_lookup/20518716*,
_output_shapes
:??????????*
dtype02'
%model_6/embedding_19/embedding_lookup?
.model_6/embedding_19/embedding_lookup/IdentityIdentity.model_6/embedding_19/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*A
_class7
53loc:@model_6/embedding_19/embedding_lookup/20518716*,
_output_shapes
:??????????20
.model_6/embedding_19/embedding_lookup/Identity?
0model_6/embedding_19/embedding_lookup/Identity_1Identity7model_6/embedding_19/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????22
0model_6/embedding_19/embedding_lookup/Identity_1?
%model_6/tf.__operators__.add_19/AddV2AddV2)model_6/tf.__operators__.add_18/AddV2:z:09model_6/embedding_19/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:??????????2'
%model_6/tf.__operators__.add_19/AddV2?
'model_6/tf.expand_dims_6/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2)
'model_6/tf.expand_dims_6/ExpandDims/dim?
#model_6/tf.expand_dims_6/ExpandDims
ExpandDimsmodel_6/tf.cast_9/Cast:y:00model_6/tf.expand_dims_6/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????2%
#model_6/tf.expand_dims_6/ExpandDims?
model_6/tf.math.multiply_6/MulMul)model_6/tf.__operators__.add_19/AddV2:z:0,model_6/tf.expand_dims_6/ExpandDims:output:0*
T0*,
_output_shapes
:??????????2 
model_6/tf.math.multiply_6/Mul?
2model_6/tf.math.reduce_sum_6/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :24
2model_6/tf.math.reduce_sum_6/Sum/reduction_indices?
 model_6/tf.math.reduce_sum_6/SumSum"model_6/tf.math.multiply_6/Mul:z:0;model_6/tf.math.reduce_sum_6/Sum/reduction_indices:output:0*
T0*(
_output_shapes
:??????????2"
 model_6/tf.math.reduce_sum_6/Sum?
model_7/flatten_7/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
model_7/flatten_7/Const?
model_7/flatten_7/ReshapeReshape
inputs_0_1 model_7/flatten_7/Const:output:0*
T0*'
_output_shapes
:?????????2
model_7/flatten_7/Reshape?
3model_7/tf.clip_by_value_10/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 * $tI25
3model_7/tf.clip_by_value_10/clip_by_value/Minimum/y?
1model_7/tf.clip_by_value_10/clip_by_value/MinimumMinimum"model_7/flatten_7/Reshape:output:0<model_7/tf.clip_by_value_10/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????23
1model_7/tf.clip_by_value_10/clip_by_value/Minimum?
+model_7/tf.clip_by_value_10/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2-
+model_7/tf.clip_by_value_10/clip_by_value/y?
)model_7/tf.clip_by_value_10/clip_by_valueMaximum5model_7/tf.clip_by_value_10/clip_by_value/Minimum:z:04model_7/tf.clip_by_value_10/clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????2+
)model_7/tf.clip_by_value_10/clip_by_value?
+model_7/tf.compat.v1.floor_div_7/FloorDiv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2-
+model_7/tf.compat.v1.floor_div_7/FloorDiv/y?
)model_7/tf.compat.v1.floor_div_7/FloorDivFloorDiv-model_7/tf.clip_by_value_10/clip_by_value:z:04model_7/tf.compat.v1.floor_div_7/FloorDiv/y:output:0*
T0*'
_output_shapes
:?????????2+
)model_7/tf.compat.v1.floor_div_7/FloorDiv?
-model_7/tf.math.greater_equal_10/GreaterEqualGreaterEqual"model_7/flatten_7/Reshape:output:0/model_7_tf_math_greater_equal_10_greaterequal_y*
T0*'
_output_shapes
:?????????2/
-model_7/tf.math.greater_equal_10/GreaterEqual?
%model_7/tf.math.floormod_7/FloorMod/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2'
%model_7/tf.math.floormod_7/FloorMod/y?
#model_7/tf.math.floormod_7/FloorModFloorMod-model_7/tf.clip_by_value_10/clip_by_value:z:0.model_7/tf.math.floormod_7/FloorMod/y:output:0*
T0*'
_output_shapes
:?????????2%
#model_7/tf.math.floormod_7/FloorMod?
model_7/embedding_23/CastCast-model_7/tf.clip_by_value_10/clip_by_value:z:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2
model_7/embedding_23/Cast?
%model_7/embedding_23/embedding_lookupResourceGather.model_7_embedding_23_embedding_lookup_20518740model_7/embedding_23/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*A
_class7
53loc:@model_7/embedding_23/embedding_lookup/20518740*,
_output_shapes
:??????????*
dtype02'
%model_7/embedding_23/embedding_lookup?
.model_7/embedding_23/embedding_lookup/IdentityIdentity.model_7/embedding_23/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*A
_class7
53loc:@model_7/embedding_23/embedding_lookup/20518740*,
_output_shapes
:??????????20
.model_7/embedding_23/embedding_lookup/Identity?
0model_7/embedding_23/embedding_lookup/Identity_1Identity7model_7/embedding_23/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????22
0model_7/embedding_23/embedding_lookup/Identity_1?
model_7/embedding_21/CastCast-model_7/tf.compat.v1.floor_div_7/FloorDiv:z:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2
model_7/embedding_21/Cast?
%model_7/embedding_21/embedding_lookupResourceGather.model_7_embedding_21_embedding_lookup_20518746model_7/embedding_21/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*A
_class7
53loc:@model_7/embedding_21/embedding_lookup/20518746*,
_output_shapes
:??????????*
dtype02'
%model_7/embedding_21/embedding_lookup?
.model_7/embedding_21/embedding_lookup/IdentityIdentity.model_7/embedding_21/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*A
_class7
53loc:@model_7/embedding_21/embedding_lookup/20518746*,
_output_shapes
:??????????20
.model_7/embedding_21/embedding_lookup/Identity?
0model_7/embedding_21/embedding_lookup/Identity_1Identity7model_7/embedding_21/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????22
0model_7/embedding_21/embedding_lookup/Identity_1?
model_7/tf.cast_10/CastCast1model_7/tf.math.greater_equal_10/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2
model_7/tf.cast_10/Cast?
%model_7/tf.__operators__.add_20/AddV2AddV29model_7/embedding_23/embedding_lookup/Identity_1:output:09model_7/embedding_21/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:??????????2'
%model_7/tf.__operators__.add_20/AddV2?
model_7/embedding_22/CastCast'model_7/tf.math.floormod_7/FloorMod:z:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2
model_7/embedding_22/Cast?
%model_7/embedding_22/embedding_lookupResourceGather.model_7_embedding_22_embedding_lookup_20518754model_7/embedding_22/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*A
_class7
53loc:@model_7/embedding_22/embedding_lookup/20518754*,
_output_shapes
:??????????*
dtype02'
%model_7/embedding_22/embedding_lookup?
.model_7/embedding_22/embedding_lookup/IdentityIdentity.model_7/embedding_22/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*A
_class7
53loc:@model_7/embedding_22/embedding_lookup/20518754*,
_output_shapes
:??????????20
.model_7/embedding_22/embedding_lookup/Identity?
0model_7/embedding_22/embedding_lookup/Identity_1Identity7model_7/embedding_22/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????22
0model_7/embedding_22/embedding_lookup/Identity_1?
%model_7/tf.__operators__.add_21/AddV2AddV2)model_7/tf.__operators__.add_20/AddV2:z:09model_7/embedding_22/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:??????????2'
%model_7/tf.__operators__.add_21/AddV2?
'model_7/tf.expand_dims_7/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2)
'model_7/tf.expand_dims_7/ExpandDims/dim?
#model_7/tf.expand_dims_7/ExpandDims
ExpandDimsmodel_7/tf.cast_10/Cast:y:00model_7/tf.expand_dims_7/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????2%
#model_7/tf.expand_dims_7/ExpandDims?
model_7/tf.math.multiply_7/MulMul)model_7/tf.__operators__.add_21/AddV2:z:0,model_7/tf.expand_dims_7/ExpandDims:output:0*
T0*,
_output_shapes
:??????????2 
model_7/tf.math.multiply_7/Mul?
2model_7/tf.math.reduce_sum_7/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :24
2model_7/tf.math.reduce_sum_7/Sum/reduction_indices?
 model_7/tf.math.reduce_sum_7/SumSum"model_7/tf.math.multiply_7/Mul:z:0;model_7/tf.math.reduce_sum_7/Sum/reduction_indices:output:0*
T0*(
_output_shapes
:??????????2"
 model_7/tf.math.reduce_sum_7/Sum?
)tf.clip_by_value_11/clip_by_value/MinimumMinimuminputs_1+tf_clip_by_value_11_clip_by_value_minimum_y*
T0*'
_output_shapes
:?????????
2+
)tf.clip_by_value_11/clip_by_value/Minimum?
!tf.clip_by_value_11/clip_by_valueMaximum-tf.clip_by_value_11/clip_by_value/Minimum:z:0#tf_clip_by_value_11_clip_by_value_y*
T0*'
_output_shapes
:?????????
2#
!tf.clip_by_value_11/clip_by_value?
tf.cast_11/CastCast)tf.math.greater_equal_11/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????
2
tf.cast_11/Castt
tf.concat_9/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
tf.concat_9/concat/axis?
tf.concat_9/concatConcatV2)model_6/tf.math.reduce_sum_6/Sum:output:0)model_7/tf.math.reduce_sum_7/Sum:output:0 tf.concat_9/concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????2
tf.concat_9/concat
tf.concat_10/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
tf.concat_10/concat/axis?
tf.concat_10/concatConcatV2%tf.clip_by_value_11/clip_by_value:z:0tf.cast_11/Cast:y:0!tf.concat_10/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????2
tf.concat_10/concat?
dense_27/MatMul/ReadVariableOpReadVariableOp'dense_27_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02 
dense_27/MatMul/ReadVariableOp?
dense_27/MatMulMatMultf.concat_9/concat:output:0&dense_27/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_27/MatMul?
dense_27/BiasAdd/ReadVariableOpReadVariableOp(dense_27_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
dense_27/BiasAdd/ReadVariableOp?
dense_27/BiasAddBiasAdddense_27/MatMul:product:0'dense_27/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_27/BiasAddt
dense_27/ReluReludense_27/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_27/Relu?
dense_30/MatMul/ReadVariableOpReadVariableOp'dense_30_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02 
dense_30/MatMul/ReadVariableOp?
dense_30/MatMulMatMultf.concat_10/concat:output:0&dense_30/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_30/MatMul?
dense_30/BiasAdd/ReadVariableOpReadVariableOp(dense_30_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
dense_30/BiasAdd/ReadVariableOp?
dense_30/BiasAddBiasAdddense_30/MatMul:product:0'dense_30/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_30/BiasAdd?
dense_28/MatMul/ReadVariableOpReadVariableOp'dense_28_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02 
dense_28/MatMul/ReadVariableOp?
dense_28/MatMulMatMuldense_27/Relu:activations:0&dense_28/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_28/MatMul?
dense_28/BiasAdd/ReadVariableOpReadVariableOp(dense_28_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
dense_28/BiasAdd/ReadVariableOp?
dense_28/BiasAddBiasAdddense_28/MatMul:product:0'dense_28/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_28/BiasAddt
dense_28/ReluReludense_28/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_28/Relu?
dense_29/MatMul/ReadVariableOpReadVariableOp'dense_29_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02 
dense_29/MatMul/ReadVariableOp?
dense_29/MatMulMatMuldense_28/Relu:activations:0&dense_29/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_29/MatMul?
dense_29/BiasAdd/ReadVariableOpReadVariableOp(dense_29_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
dense_29/BiasAdd/ReadVariableOp?
dense_29/BiasAddBiasAdddense_29/MatMul:product:0'dense_29/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_29/BiasAddt
dense_29/ReluReludense_29/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_29/Relu?
dense_31/MatMul/ReadVariableOpReadVariableOp'dense_31_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02 
dense_31/MatMul/ReadVariableOp?
dense_31/MatMulMatMuldense_30/BiasAdd:output:0&dense_31/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_31/MatMul?
dense_31/BiasAdd/ReadVariableOpReadVariableOp(dense_31_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
dense_31/BiasAdd/ReadVariableOp?
dense_31/BiasAddBiasAdddense_31/MatMul:product:0'dense_31/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_31/BiasAdd
tf.concat_11/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
tf.concat_11/concat/axis?
tf.concat_11/concatConcatV2dense_29/Relu:activations:0dense_31/BiasAdd:output:0!tf.concat_11/concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????2
tf.concat_11/concat?
dense_32/MatMul/ReadVariableOpReadVariableOp'dense_32_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02 
dense_32/MatMul/ReadVariableOp?
dense_32/MatMulMatMultf.concat_11/concat:output:0&dense_32/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_32/MatMul?
dense_32/BiasAdd/ReadVariableOpReadVariableOp(dense_32_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
dense_32/BiasAdd/ReadVariableOp?
dense_32/BiasAddBiasAdddense_32/MatMul:product:0'dense_32/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_32/BiasAdd|
tf.nn.relu_9/ReluReludense_32/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
tf.nn.relu_9/Relu?
dense_33/MatMul/ReadVariableOpReadVariableOp'dense_33_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02 
dense_33/MatMul/ReadVariableOp?
dense_33/MatMulMatMultf.nn.relu_9/Relu:activations:0&dense_33/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_33/MatMul?
dense_33/BiasAdd/ReadVariableOpReadVariableOp(dense_33_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
dense_33/BiasAdd/ReadVariableOp?
dense_33/BiasAddBiasAdddense_33/MatMul:product:0'dense_33/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_33/BiasAdd?
tf.__operators__.add_22/AddV2AddV2dense_33/BiasAdd:output:0tf.nn.relu_9/Relu:activations:0*
T0*(
_output_shapes
:??????????2
tf.__operators__.add_22/AddV2?
tf.nn.relu_10/ReluRelu!tf.__operators__.add_22/AddV2:z:0*
T0*(
_output_shapes
:??????????2
tf.nn.relu_10/Relu?
dense_34/MatMul/ReadVariableOpReadVariableOp'dense_34_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02 
dense_34/MatMul/ReadVariableOp?
dense_34/MatMulMatMul tf.nn.relu_10/Relu:activations:0&dense_34/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_34/MatMul?
dense_34/BiasAdd/ReadVariableOpReadVariableOp(dense_34_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
dense_34/BiasAdd/ReadVariableOp?
dense_34/BiasAddBiasAdddense_34/MatMul:product:0'dense_34/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_34/BiasAdd?
tf.__operators__.add_23/AddV2AddV2dense_34/BiasAdd:output:0 tf.nn.relu_10/Relu:activations:0*
T0*(
_output_shapes
:??????????2
tf.__operators__.add_23/AddV2?
tf.nn.relu_11/ReluRelu!tf.__operators__.add_23/AddV2:z:0*
T0*(
_output_shapes
:??????????2
tf.nn.relu_11/Relu?
2normalize_3/normalization_3/Reshape/ReadVariableOpReadVariableOp;normalize_3_normalization_3_reshape_readvariableop_resource*
_output_shapes	
:?*
dtype024
2normalize_3/normalization_3/Reshape/ReadVariableOp?
)normalize_3/normalization_3/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2+
)normalize_3/normalization_3/Reshape/shape?
#normalize_3/normalization_3/ReshapeReshape:normalize_3/normalization_3/Reshape/ReadVariableOp:value:02normalize_3/normalization_3/Reshape/shape:output:0*
T0*
_output_shapes
:	?2%
#normalize_3/normalization_3/Reshape?
4normalize_3/normalization_3/Reshape_1/ReadVariableOpReadVariableOp=normalize_3_normalization_3_reshape_1_readvariableop_resource*
_output_shapes	
:?*
dtype026
4normalize_3/normalization_3/Reshape_1/ReadVariableOp?
+normalize_3/normalization_3/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2-
+normalize_3/normalization_3/Reshape_1/shape?
%normalize_3/normalization_3/Reshape_1Reshape<normalize_3/normalization_3/Reshape_1/ReadVariableOp:value:04normalize_3/normalization_3/Reshape_1/shape:output:0*
T0*
_output_shapes
:	?2'
%normalize_3/normalization_3/Reshape_1?
normalize_3/normalization_3/subSub tf.nn.relu_11/Relu:activations:0,normalize_3/normalization_3/Reshape:output:0*
T0*(
_output_shapes
:??????????2!
normalize_3/normalization_3/sub?
 normalize_3/normalization_3/SqrtSqrt.normalize_3/normalization_3/Reshape_1:output:0*
T0*
_output_shapes
:	?2"
 normalize_3/normalization_3/Sqrt?
%normalize_3/normalization_3/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32'
%normalize_3/normalization_3/Maximum/y?
#normalize_3/normalization_3/MaximumMaximum$normalize_3/normalization_3/Sqrt:y:0.normalize_3/normalization_3/Maximum/y:output:0*
T0*
_output_shapes
:	?2%
#normalize_3/normalization_3/Maximum?
#normalize_3/normalization_3/truedivRealDiv#normalize_3/normalization_3/sub:z:0'normalize_3/normalization_3/Maximum:z:0*
T0*(
_output_shapes
:??????????2%
#normalize_3/normalization_3/truediv?
dense_35/MatMul/ReadVariableOpReadVariableOp'dense_35_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02 
dense_35/MatMul/ReadVariableOp?
dense_35/MatMulMatMul'normalize_3/normalization_3/truediv:z:0&dense_35/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_35/MatMul?
dense_35/BiasAdd/ReadVariableOpReadVariableOp(dense_35_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_35/BiasAdd/ReadVariableOp?
dense_35/BiasAddBiasAdddense_35/MatMul:product:0'dense_35/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_35/BiasAdd?
IdentityIdentitydense_35/BiasAdd:output:0 ^dense_27/BiasAdd/ReadVariableOp^dense_27/MatMul/ReadVariableOp ^dense_28/BiasAdd/ReadVariableOp^dense_28/MatMul/ReadVariableOp ^dense_29/BiasAdd/ReadVariableOp^dense_29/MatMul/ReadVariableOp ^dense_30/BiasAdd/ReadVariableOp^dense_30/MatMul/ReadVariableOp ^dense_31/BiasAdd/ReadVariableOp^dense_31/MatMul/ReadVariableOp ^dense_32/BiasAdd/ReadVariableOp^dense_32/MatMul/ReadVariableOp ^dense_33/BiasAdd/ReadVariableOp^dense_33/MatMul/ReadVariableOp ^dense_34/BiasAdd/ReadVariableOp^dense_34/MatMul/ReadVariableOp ^dense_35/BiasAdd/ReadVariableOp^dense_35/MatMul/ReadVariableOp&^model_6/embedding_18/embedding_lookup&^model_6/embedding_19/embedding_lookup&^model_6/embedding_20/embedding_lookup&^model_7/embedding_21/embedding_lookup&^model_7/embedding_22/embedding_lookup&^model_7/embedding_23/embedding_lookup3^normalize_3/normalization_3/Reshape/ReadVariableOp5^normalize_3/normalization_3/Reshape_1/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????:?????????:?????????
: : :::: :::: : ::::::::::::::::::::2B
dense_27/BiasAdd/ReadVariableOpdense_27/BiasAdd/ReadVariableOp2@
dense_27/MatMul/ReadVariableOpdense_27/MatMul/ReadVariableOp2B
dense_28/BiasAdd/ReadVariableOpdense_28/BiasAdd/ReadVariableOp2@
dense_28/MatMul/ReadVariableOpdense_28/MatMul/ReadVariableOp2B
dense_29/BiasAdd/ReadVariableOpdense_29/BiasAdd/ReadVariableOp2@
dense_29/MatMul/ReadVariableOpdense_29/MatMul/ReadVariableOp2B
dense_30/BiasAdd/ReadVariableOpdense_30/BiasAdd/ReadVariableOp2@
dense_30/MatMul/ReadVariableOpdense_30/MatMul/ReadVariableOp2B
dense_31/BiasAdd/ReadVariableOpdense_31/BiasAdd/ReadVariableOp2@
dense_31/MatMul/ReadVariableOpdense_31/MatMul/ReadVariableOp2B
dense_32/BiasAdd/ReadVariableOpdense_32/BiasAdd/ReadVariableOp2@
dense_32/MatMul/ReadVariableOpdense_32/MatMul/ReadVariableOp2B
dense_33/BiasAdd/ReadVariableOpdense_33/BiasAdd/ReadVariableOp2@
dense_33/MatMul/ReadVariableOpdense_33/MatMul/ReadVariableOp2B
dense_34/BiasAdd/ReadVariableOpdense_34/BiasAdd/ReadVariableOp2@
dense_34/MatMul/ReadVariableOpdense_34/MatMul/ReadVariableOp2B
dense_35/BiasAdd/ReadVariableOpdense_35/BiasAdd/ReadVariableOp2@
dense_35/MatMul/ReadVariableOpdense_35/MatMul/ReadVariableOp2N
%model_6/embedding_18/embedding_lookup%model_6/embedding_18/embedding_lookup2N
%model_6/embedding_19/embedding_lookup%model_6/embedding_19/embedding_lookup2N
%model_6/embedding_20/embedding_lookup%model_6/embedding_20/embedding_lookup2N
%model_7/embedding_21/embedding_lookup%model_7/embedding_21/embedding_lookup2N
%model_7/embedding_22/embedding_lookup%model_7/embedding_22/embedding_lookup2N
%model_7/embedding_23/embedding_lookup%model_7/embedding_23/embedding_lookup2h
2normalize_3/normalization_3/Reshape/ReadVariableOp2normalize_3/normalization_3/Reshape/ReadVariableOp2l
4normalize_3/normalization_3/Reshape_1/ReadVariableOp4normalize_3/normalization_3/Reshape_1/ReadVariableOp:S O
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
?
?
*__inference_model_6_layer_call_fn_20517552
input_7
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_7unknown	unknown_0	unknown_1	unknown_2*
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
GPU2 *0J 8? *N
fIRG
E__inference_model_6_layer_call_and_return_conditional_losses_205175412
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:?????????: :::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_7:

_output_shapes
: 
?
?
1__inference_custom_model_3_layer_call_fn_20519091

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
L__inference_custom_model_3_layer_call_and_return_conditional_losses_205183772
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
J__inference_embedding_23_layer_call_and_return_conditional_losses_20519663

inputs
embedding_lookup_20519657
identity??embedding_lookup]
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:?????????2
Cast?
embedding_lookupResourceGatherembedding_lookup_20519657Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*,
_class"
 loc:@embedding_lookup/20519657*,
_output_shapes
:??????????*
dtype02
embedding_lookup?
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*,
_class"
 loc:@embedding_lookup/20519657*,
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
F__inference_dense_34_layer_call_and_return_conditional_losses_20519526

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
/__inference_embedding_18_layer_call_fn_20519625

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
J__inference_embedding_18_layer_call_and_return_conditional_losses_205174312
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
F__inference_dense_32_layer_call_and_return_conditional_losses_20519488

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
?
?
*__inference_model_7_layer_call_fn_20519380

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
GPU2 *0J 8? *N
fIRG
E__inference_model_7_layer_call_and_return_conditional_losses_205178122
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
J__inference_embedding_22_layer_call_and_return_conditional_losses_20517681

inputs
embedding_lookup_20517675
identity??embedding_lookup]
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:?????????2
Cast?
embedding_lookupResourceGatherembedding_lookup_20517675Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*,
_class"
 loc:@embedding_lookup/20517675*,
_output_shapes
:??????????*
dtype02
embedding_lookup?
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*,
_class"
 loc:@embedding_lookup/20517675*,
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
F__inference_dense_34_layer_call_and_return_conditional_losses_20518110

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
*__inference_model_7_layer_call_fn_20517778
input_8
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_8unknown	unknown_0	unknown_1	unknown_2*
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
GPU2 *0J 8? *N
fIRG
E__inference_model_7_layer_call_and_return_conditional_losses_205177672
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:?????????: :::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_8:

_output_shapes
: 
?	
?
J__inference_embedding_23_layer_call_and_return_conditional_losses_20517635

inputs
embedding_lookup_20517629
identity??embedding_lookup]
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:?????????2
Cast?
embedding_lookupResourceGatherembedding_lookup_20517629Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*,
_class"
 loc:@embedding_lookup/20517629*,
_output_shapes
:??????????*
dtype02
embedding_lookup?
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*,
_class"
 loc:@embedding_lookup/20517629*,
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
?
?
+__inference_dense_28_layer_call_fn_20519420

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
F__inference_dense_28_layer_call_and_return_conditional_losses_205179742
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
J__inference_embedding_20_layer_call_and_return_conditional_losses_20517409

inputs
embedding_lookup_20517403
identity??embedding_lookup]
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:?????????2
Cast?
embedding_lookupResourceGatherembedding_lookup_20517403Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*,
_class"
 loc:@embedding_lookup/20517403*,
_output_shapes
:??????????*
dtype02
embedding_lookup?
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*,
_class"
 loc:@embedding_lookup/20517403*,
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
?
?
+__inference_dense_32_layer_call_fn_20519497

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
F__inference_dense_32_layer_call_and_return_conditional_losses_205180552
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
?	
?
J__inference_embedding_18_layer_call_and_return_conditional_losses_20517431

inputs
embedding_lookup_20517425
identity??embedding_lookup]
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:?????????2
Cast?
embedding_lookupResourceGatherembedding_lookup_20517425Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*,
_class"
 loc:@embedding_lookup/20517425*,
_output_shapes
:??????????*
dtype02
embedding_lookup?
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*,
_class"
 loc:@embedding_lookup/20517425*,
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
?Y
?

L__inference_custom_model_3_layer_call_and_return_conditional_losses_20518377

inputs
inputs_1
inputs_2+
'tf_math_greater_equal_11_greaterequal_y
model_6_20518292
model_6_20518294
model_6_20518296
model_6_20518298
model_7_20518301
model_7_20518303
model_7_20518305
model_7_20518307/
+tf_clip_by_value_11_clip_by_value_minimum_y'
#tf_clip_by_value_11_clip_by_value_y
dense_27_20518319
dense_27_20518321
dense_30_20518324
dense_30_20518326
dense_28_20518329
dense_28_20518331
dense_29_20518334
dense_29_20518336
dense_31_20518339
dense_31_20518341
dense_32_20518346
dense_32_20518348
dense_33_20518352
dense_33_20518354
dense_34_20518359
dense_34_20518361
normalize_3_20518366
normalize_3_20518368
dense_35_20518371
dense_35_20518373
identity?? dense_27/StatefulPartitionedCall? dense_28/StatefulPartitionedCall? dense_29/StatefulPartitionedCall? dense_30/StatefulPartitionedCall? dense_31/StatefulPartitionedCall? dense_32/StatefulPartitionedCall? dense_33/StatefulPartitionedCall? dense_34/StatefulPartitionedCall? dense_35/StatefulPartitionedCall?model_6/StatefulPartitionedCall?model_7/StatefulPartitionedCall?#normalize_3/StatefulPartitionedCall?
%tf.math.greater_equal_11/GreaterEqualGreaterEqualinputs_2'tf_math_greater_equal_11_greaterequal_y*
T0*'
_output_shapes
:?????????
2'
%tf.math.greater_equal_11/GreaterEqual?
model_6/StatefulPartitionedCallStatefulPartitionedCallinputsmodel_6_20518292model_6_20518294model_6_20518296model_6_20518298*
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
GPU2 *0J 8? *N
fIRG
E__inference_model_6_layer_call_and_return_conditional_losses_205175412!
model_6/StatefulPartitionedCall?
model_7/StatefulPartitionedCallStatefulPartitionedCallinputs_1model_7_20518301model_7_20518303model_7_20518305model_7_20518307*
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
GPU2 *0J 8? *N
fIRG
E__inference_model_7_layer_call_and_return_conditional_losses_205177672!
model_7/StatefulPartitionedCall?
)tf.clip_by_value_11/clip_by_value/MinimumMinimuminputs_2+tf_clip_by_value_11_clip_by_value_minimum_y*
T0*'
_output_shapes
:?????????
2+
)tf.clip_by_value_11/clip_by_value/Minimum?
!tf.clip_by_value_11/clip_by_valueMaximum-tf.clip_by_value_11/clip_by_value/Minimum:z:0#tf_clip_by_value_11_clip_by_value_y*
T0*'
_output_shapes
:?????????
2#
!tf.clip_by_value_11/clip_by_value?
tf.cast_11/CastCast)tf.math.greater_equal_11/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????
2
tf.cast_11/Castt
tf.concat_9/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
tf.concat_9/concat/axis?
tf.concat_9/concatConcatV2(model_6/StatefulPartitionedCall:output:0(model_7/StatefulPartitionedCall:output:0 tf.concat_9/concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????2
tf.concat_9/concat
tf.concat_10/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
tf.concat_10/concat/axis?
tf.concat_10/concatConcatV2%tf.clip_by_value_11/clip_by_value:z:0tf.cast_11/Cast:y:0!tf.concat_10/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????2
tf.concat_10/concat?
 dense_27/StatefulPartitionedCallStatefulPartitionedCalltf.concat_9/concat:output:0dense_27_20518319dense_27_20518321*
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
F__inference_dense_27_layer_call_and_return_conditional_losses_205179212"
 dense_27/StatefulPartitionedCall?
 dense_30/StatefulPartitionedCallStatefulPartitionedCalltf.concat_10/concat:output:0dense_30_20518324dense_30_20518326*
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
F__inference_dense_30_layer_call_and_return_conditional_losses_205179472"
 dense_30/StatefulPartitionedCall?
 dense_28/StatefulPartitionedCallStatefulPartitionedCall)dense_27/StatefulPartitionedCall:output:0dense_28_20518329dense_28_20518331*
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
F__inference_dense_28_layer_call_and_return_conditional_losses_205179742"
 dense_28/StatefulPartitionedCall?
 dense_29/StatefulPartitionedCallStatefulPartitionedCall)dense_28/StatefulPartitionedCall:output:0dense_29_20518334dense_29_20518336*
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
F__inference_dense_29_layer_call_and_return_conditional_losses_205180012"
 dense_29/StatefulPartitionedCall?
 dense_31/StatefulPartitionedCallStatefulPartitionedCall)dense_30/StatefulPartitionedCall:output:0dense_31_20518339dense_31_20518341*
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
F__inference_dense_31_layer_call_and_return_conditional_losses_205180272"
 dense_31/StatefulPartitionedCall
tf.concat_11/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
tf.concat_11/concat/axis?
tf.concat_11/concatConcatV2)dense_29/StatefulPartitionedCall:output:0)dense_31/StatefulPartitionedCall:output:0!tf.concat_11/concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????2
tf.concat_11/concat?
 dense_32/StatefulPartitionedCallStatefulPartitionedCalltf.concat_11/concat:output:0dense_32_20518346dense_32_20518348*
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
F__inference_dense_32_layer_call_and_return_conditional_losses_205180552"
 dense_32/StatefulPartitionedCall?
tf.nn.relu_9/ReluRelu)dense_32/StatefulPartitionedCall:output:0*
T0*(
_output_shapes
:??????????2
tf.nn.relu_9/Relu?
 dense_33/StatefulPartitionedCallStatefulPartitionedCalltf.nn.relu_9/Relu:activations:0dense_33_20518352dense_33_20518354*
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
F__inference_dense_33_layer_call_and_return_conditional_losses_205180822"
 dense_33/StatefulPartitionedCall?
tf.__operators__.add_22/AddV2AddV2)dense_33/StatefulPartitionedCall:output:0tf.nn.relu_9/Relu:activations:0*
T0*(
_output_shapes
:??????????2
tf.__operators__.add_22/AddV2?
tf.nn.relu_10/ReluRelu!tf.__operators__.add_22/AddV2:z:0*
T0*(
_output_shapes
:??????????2
tf.nn.relu_10/Relu?
 dense_34/StatefulPartitionedCallStatefulPartitionedCall tf.nn.relu_10/Relu:activations:0dense_34_20518359dense_34_20518361*
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
F__inference_dense_34_layer_call_and_return_conditional_losses_205181102"
 dense_34/StatefulPartitionedCall?
tf.__operators__.add_23/AddV2AddV2)dense_34/StatefulPartitionedCall:output:0 tf.nn.relu_10/Relu:activations:0*
T0*(
_output_shapes
:??????????2
tf.__operators__.add_23/AddV2?
tf.nn.relu_11/ReluRelu!tf.__operators__.add_23/AddV2:z:0*
T0*(
_output_shapes
:??????????2
tf.nn.relu_11/Relu?
#normalize_3/StatefulPartitionedCallStatefulPartitionedCall tf.nn.relu_11/Relu:activations:0normalize_3_20518366normalize_3_20518368*
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
I__inference_normalize_3_layer_call_and_return_conditional_losses_205181452%
#normalize_3/StatefulPartitionedCall?
 dense_35/StatefulPartitionedCallStatefulPartitionedCall,normalize_3/StatefulPartitionedCall:output:0dense_35_20518371dense_35_20518373*
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
F__inference_dense_35_layer_call_and_return_conditional_losses_205181712"
 dense_35/StatefulPartitionedCall?
IdentityIdentity)dense_35/StatefulPartitionedCall:output:0!^dense_27/StatefulPartitionedCall!^dense_28/StatefulPartitionedCall!^dense_29/StatefulPartitionedCall!^dense_30/StatefulPartitionedCall!^dense_31/StatefulPartitionedCall!^dense_32/StatefulPartitionedCall!^dense_33/StatefulPartitionedCall!^dense_34/StatefulPartitionedCall!^dense_35/StatefulPartitionedCall ^model_6/StatefulPartitionedCall ^model_7/StatefulPartitionedCall$^normalize_3/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????:?????????:?????????
: : :::: :::: : ::::::::::::::::::::2D
 dense_27/StatefulPartitionedCall dense_27/StatefulPartitionedCall2D
 dense_28/StatefulPartitionedCall dense_28/StatefulPartitionedCall2D
 dense_29/StatefulPartitionedCall dense_29/StatefulPartitionedCall2D
 dense_30/StatefulPartitionedCall dense_30/StatefulPartitionedCall2D
 dense_31/StatefulPartitionedCall dense_31/StatefulPartitionedCall2D
 dense_32/StatefulPartitionedCall dense_32/StatefulPartitionedCall2D
 dense_33/StatefulPartitionedCall dense_33/StatefulPartitionedCall2D
 dense_34/StatefulPartitionedCall dense_34/StatefulPartitionedCall2D
 dense_35/StatefulPartitionedCall dense_35/StatefulPartitionedCall2B
model_6/StatefulPartitionedCallmodel_6/StatefulPartitionedCall2B
model_7/StatefulPartitionedCallmodel_7/StatefulPartitionedCall2J
#normalize_3/StatefulPartitionedCall#normalize_3/StatefulPartitionedCall:O K
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
?
?
I__inference_normalize_3_layer_call_and_return_conditional_losses_20518145
x3
/normalization_3_reshape_readvariableop_resource5
1normalization_3_reshape_1_readvariableop_resource
identity??&normalization_3/Reshape/ReadVariableOp?(normalization_3/Reshape_1/ReadVariableOp?
&normalization_3/Reshape/ReadVariableOpReadVariableOp/normalization_3_reshape_readvariableop_resource*
_output_shapes	
:?*
dtype02(
&normalization_3/Reshape/ReadVariableOp?
normalization_3/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2
normalization_3/Reshape/shape?
normalization_3/ReshapeReshape.normalization_3/Reshape/ReadVariableOp:value:0&normalization_3/Reshape/shape:output:0*
T0*
_output_shapes
:	?2
normalization_3/Reshape?
(normalization_3/Reshape_1/ReadVariableOpReadVariableOp1normalization_3_reshape_1_readvariableop_resource*
_output_shapes	
:?*
dtype02*
(normalization_3/Reshape_1/ReadVariableOp?
normalization_3/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2!
normalization_3/Reshape_1/shape?
normalization_3/Reshape_1Reshape0normalization_3/Reshape_1/ReadVariableOp:value:0(normalization_3/Reshape_1/shape:output:0*
T0*
_output_shapes
:	?2
normalization_3/Reshape_1?
normalization_3/subSubx normalization_3/Reshape:output:0*
T0*(
_output_shapes
:??????????2
normalization_3/sub?
normalization_3/SqrtSqrt"normalization_3/Reshape_1:output:0*
T0*
_output_shapes
:	?2
normalization_3/Sqrt{
normalization_3/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32
normalization_3/Maximum/y?
normalization_3/MaximumMaximumnormalization_3/Sqrt:y:0"normalization_3/Maximum/y:output:0*
T0*
_output_shapes
:	?2
normalization_3/Maximum?
normalization_3/truedivRealDivnormalization_3/sub:z:0normalization_3/Maximum:z:0*
T0*(
_output_shapes
:??????????2
normalization_3/truediv?
IdentityIdentitynormalization_3/truediv:z:0'^normalization_3/Reshape/ReadVariableOp)^normalization_3/Reshape_1/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::2P
&normalization_3/Reshape/ReadVariableOp&normalization_3/Reshape/ReadVariableOp2T
(normalization_3/Reshape_1/ReadVariableOp(normalization_3/Reshape_1/ReadVariableOp:K G
(
_output_shapes
:??????????

_user_specified_namex
?	
?
F__inference_dense_31_layer_call_and_return_conditional_losses_20519469

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
J__inference_embedding_19_layer_call_and_return_conditional_losses_20517455

inputs
embedding_lookup_20517449
identity??embedding_lookup]
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:?????????2
Cast?
embedding_lookupResourceGatherembedding_lookup_20517449Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*,
_class"
 loc:@embedding_lookup/20517449*,
_output_shapes
:??????????*
dtype02
embedding_lookup?
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*,
_class"
 loc:@embedding_lookup/20517449*,
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
F__inference_dense_30_layer_call_and_return_conditional_losses_20517947

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
޺
?
#__inference__wrapped_model_20517371

cards0

cards1
bets:
6custom_model_3_tf_math_greater_equal_11_greaterequal_yA
=custom_model_3_model_6_tf_math_greater_equal_9_greaterequal_yA
=custom_model_3_model_6_embedding_20_embedding_lookup_20517221A
=custom_model_3_model_6_embedding_18_embedding_lookup_20517227A
=custom_model_3_model_6_embedding_19_embedding_lookup_20517235B
>custom_model_3_model_7_tf_math_greater_equal_10_greaterequal_yA
=custom_model_3_model_7_embedding_23_embedding_lookup_20517259A
=custom_model_3_model_7_embedding_21_embedding_lookup_20517265A
=custom_model_3_model_7_embedding_22_embedding_lookup_20517273>
:custom_model_3_tf_clip_by_value_11_clip_by_value_minimum_y6
2custom_model_3_tf_clip_by_value_11_clip_by_value_y:
6custom_model_3_dense_27_matmul_readvariableop_resource;
7custom_model_3_dense_27_biasadd_readvariableop_resource:
6custom_model_3_dense_30_matmul_readvariableop_resource;
7custom_model_3_dense_30_biasadd_readvariableop_resource:
6custom_model_3_dense_28_matmul_readvariableop_resource;
7custom_model_3_dense_28_biasadd_readvariableop_resource:
6custom_model_3_dense_29_matmul_readvariableop_resource;
7custom_model_3_dense_29_biasadd_readvariableop_resource:
6custom_model_3_dense_31_matmul_readvariableop_resource;
7custom_model_3_dense_31_biasadd_readvariableop_resource:
6custom_model_3_dense_32_matmul_readvariableop_resource;
7custom_model_3_dense_32_biasadd_readvariableop_resource:
6custom_model_3_dense_33_matmul_readvariableop_resource;
7custom_model_3_dense_33_biasadd_readvariableop_resource:
6custom_model_3_dense_34_matmul_readvariableop_resource;
7custom_model_3_dense_34_biasadd_readvariableop_resourceN
Jcustom_model_3_normalize_3_normalization_3_reshape_readvariableop_resourceP
Lcustom_model_3_normalize_3_normalization_3_reshape_1_readvariableop_resource:
6custom_model_3_dense_35_matmul_readvariableop_resource;
7custom_model_3_dense_35_biasadd_readvariableop_resource
identity??.custom_model_3/dense_27/BiasAdd/ReadVariableOp?-custom_model_3/dense_27/MatMul/ReadVariableOp?.custom_model_3/dense_28/BiasAdd/ReadVariableOp?-custom_model_3/dense_28/MatMul/ReadVariableOp?.custom_model_3/dense_29/BiasAdd/ReadVariableOp?-custom_model_3/dense_29/MatMul/ReadVariableOp?.custom_model_3/dense_30/BiasAdd/ReadVariableOp?-custom_model_3/dense_30/MatMul/ReadVariableOp?.custom_model_3/dense_31/BiasAdd/ReadVariableOp?-custom_model_3/dense_31/MatMul/ReadVariableOp?.custom_model_3/dense_32/BiasAdd/ReadVariableOp?-custom_model_3/dense_32/MatMul/ReadVariableOp?.custom_model_3/dense_33/BiasAdd/ReadVariableOp?-custom_model_3/dense_33/MatMul/ReadVariableOp?.custom_model_3/dense_34/BiasAdd/ReadVariableOp?-custom_model_3/dense_34/MatMul/ReadVariableOp?.custom_model_3/dense_35/BiasAdd/ReadVariableOp?-custom_model_3/dense_35/MatMul/ReadVariableOp?4custom_model_3/model_6/embedding_18/embedding_lookup?4custom_model_3/model_6/embedding_19/embedding_lookup?4custom_model_3/model_6/embedding_20/embedding_lookup?4custom_model_3/model_7/embedding_21/embedding_lookup?4custom_model_3/model_7/embedding_22/embedding_lookup?4custom_model_3/model_7/embedding_23/embedding_lookup?Acustom_model_3/normalize_3/normalization_3/Reshape/ReadVariableOp?Ccustom_model_3/normalize_3/normalization_3/Reshape_1/ReadVariableOp?
4custom_model_3/tf.math.greater_equal_11/GreaterEqualGreaterEqualbets6custom_model_3_tf_math_greater_equal_11_greaterequal_y*
T0*'
_output_shapes
:?????????
26
4custom_model_3/tf.math.greater_equal_11/GreaterEqual?
&custom_model_3/model_6/flatten_6/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2(
&custom_model_3/model_6/flatten_6/Const?
(custom_model_3/model_6/flatten_6/ReshapeReshapecards0/custom_model_3/model_6/flatten_6/Const:output:0*
T0*'
_output_shapes
:?????????2*
(custom_model_3/model_6/flatten_6/Reshape?
Acustom_model_3/model_6/tf.clip_by_value_9/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 * $tI2C
Acustom_model_3/model_6/tf.clip_by_value_9/clip_by_value/Minimum/y?
?custom_model_3/model_6/tf.clip_by_value_9/clip_by_value/MinimumMinimum1custom_model_3/model_6/flatten_6/Reshape:output:0Jcustom_model_3/model_6/tf.clip_by_value_9/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2A
?custom_model_3/model_6/tf.clip_by_value_9/clip_by_value/Minimum?
9custom_model_3/model_6/tf.clip_by_value_9/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2;
9custom_model_3/model_6/tf.clip_by_value_9/clip_by_value/y?
7custom_model_3/model_6/tf.clip_by_value_9/clip_by_valueMaximumCcustom_model_3/model_6/tf.clip_by_value_9/clip_by_value/Minimum:z:0Bcustom_model_3/model_6/tf.clip_by_value_9/clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????29
7custom_model_3/model_6/tf.clip_by_value_9/clip_by_value?
:custom_model_3/model_6/tf.compat.v1.floor_div_6/FloorDiv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2<
:custom_model_3/model_6/tf.compat.v1.floor_div_6/FloorDiv/y?
8custom_model_3/model_6/tf.compat.v1.floor_div_6/FloorDivFloorDiv;custom_model_3/model_6/tf.clip_by_value_9/clip_by_value:z:0Ccustom_model_3/model_6/tf.compat.v1.floor_div_6/FloorDiv/y:output:0*
T0*'
_output_shapes
:?????????2:
8custom_model_3/model_6/tf.compat.v1.floor_div_6/FloorDiv?
;custom_model_3/model_6/tf.math.greater_equal_9/GreaterEqualGreaterEqual1custom_model_3/model_6/flatten_6/Reshape:output:0=custom_model_3_model_6_tf_math_greater_equal_9_greaterequal_y*
T0*'
_output_shapes
:?????????2=
;custom_model_3/model_6/tf.math.greater_equal_9/GreaterEqual?
4custom_model_3/model_6/tf.math.floormod_6/FloorMod/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@26
4custom_model_3/model_6/tf.math.floormod_6/FloorMod/y?
2custom_model_3/model_6/tf.math.floormod_6/FloorModFloorMod;custom_model_3/model_6/tf.clip_by_value_9/clip_by_value:z:0=custom_model_3/model_6/tf.math.floormod_6/FloorMod/y:output:0*
T0*'
_output_shapes
:?????????24
2custom_model_3/model_6/tf.math.floormod_6/FloorMod?
(custom_model_3/model_6/embedding_20/CastCast;custom_model_3/model_6/tf.clip_by_value_9/clip_by_value:z:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2*
(custom_model_3/model_6/embedding_20/Cast?
4custom_model_3/model_6/embedding_20/embedding_lookupResourceGather=custom_model_3_model_6_embedding_20_embedding_lookup_20517221,custom_model_3/model_6/embedding_20/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*P
_classF
DBloc:@custom_model_3/model_6/embedding_20/embedding_lookup/20517221*,
_output_shapes
:??????????*
dtype026
4custom_model_3/model_6/embedding_20/embedding_lookup?
=custom_model_3/model_6/embedding_20/embedding_lookup/IdentityIdentity=custom_model_3/model_6/embedding_20/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*P
_classF
DBloc:@custom_model_3/model_6/embedding_20/embedding_lookup/20517221*,
_output_shapes
:??????????2?
=custom_model_3/model_6/embedding_20/embedding_lookup/Identity?
?custom_model_3/model_6/embedding_20/embedding_lookup/Identity_1IdentityFcustom_model_3/model_6/embedding_20/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????2A
?custom_model_3/model_6/embedding_20/embedding_lookup/Identity_1?
(custom_model_3/model_6/embedding_18/CastCast<custom_model_3/model_6/tf.compat.v1.floor_div_6/FloorDiv:z:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2*
(custom_model_3/model_6/embedding_18/Cast?
4custom_model_3/model_6/embedding_18/embedding_lookupResourceGather=custom_model_3_model_6_embedding_18_embedding_lookup_20517227,custom_model_3/model_6/embedding_18/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*P
_classF
DBloc:@custom_model_3/model_6/embedding_18/embedding_lookup/20517227*,
_output_shapes
:??????????*
dtype026
4custom_model_3/model_6/embedding_18/embedding_lookup?
=custom_model_3/model_6/embedding_18/embedding_lookup/IdentityIdentity=custom_model_3/model_6/embedding_18/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*P
_classF
DBloc:@custom_model_3/model_6/embedding_18/embedding_lookup/20517227*,
_output_shapes
:??????????2?
=custom_model_3/model_6/embedding_18/embedding_lookup/Identity?
?custom_model_3/model_6/embedding_18/embedding_lookup/Identity_1IdentityFcustom_model_3/model_6/embedding_18/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????2A
?custom_model_3/model_6/embedding_18/embedding_lookup/Identity_1?
%custom_model_3/model_6/tf.cast_9/CastCast?custom_model_3/model_6/tf.math.greater_equal_9/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2'
%custom_model_3/model_6/tf.cast_9/Cast?
4custom_model_3/model_6/tf.__operators__.add_18/AddV2AddV2Hcustom_model_3/model_6/embedding_20/embedding_lookup/Identity_1:output:0Hcustom_model_3/model_6/embedding_18/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:??????????26
4custom_model_3/model_6/tf.__operators__.add_18/AddV2?
(custom_model_3/model_6/embedding_19/CastCast6custom_model_3/model_6/tf.math.floormod_6/FloorMod:z:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2*
(custom_model_3/model_6/embedding_19/Cast?
4custom_model_3/model_6/embedding_19/embedding_lookupResourceGather=custom_model_3_model_6_embedding_19_embedding_lookup_20517235,custom_model_3/model_6/embedding_19/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*P
_classF
DBloc:@custom_model_3/model_6/embedding_19/embedding_lookup/20517235*,
_output_shapes
:??????????*
dtype026
4custom_model_3/model_6/embedding_19/embedding_lookup?
=custom_model_3/model_6/embedding_19/embedding_lookup/IdentityIdentity=custom_model_3/model_6/embedding_19/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*P
_classF
DBloc:@custom_model_3/model_6/embedding_19/embedding_lookup/20517235*,
_output_shapes
:??????????2?
=custom_model_3/model_6/embedding_19/embedding_lookup/Identity?
?custom_model_3/model_6/embedding_19/embedding_lookup/Identity_1IdentityFcustom_model_3/model_6/embedding_19/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????2A
?custom_model_3/model_6/embedding_19/embedding_lookup/Identity_1?
4custom_model_3/model_6/tf.__operators__.add_19/AddV2AddV28custom_model_3/model_6/tf.__operators__.add_18/AddV2:z:0Hcustom_model_3/model_6/embedding_19/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:??????????26
4custom_model_3/model_6/tf.__operators__.add_19/AddV2?
6custom_model_3/model_6/tf.expand_dims_6/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????28
6custom_model_3/model_6/tf.expand_dims_6/ExpandDims/dim?
2custom_model_3/model_6/tf.expand_dims_6/ExpandDims
ExpandDims)custom_model_3/model_6/tf.cast_9/Cast:y:0?custom_model_3/model_6/tf.expand_dims_6/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????24
2custom_model_3/model_6/tf.expand_dims_6/ExpandDims?
-custom_model_3/model_6/tf.math.multiply_6/MulMul8custom_model_3/model_6/tf.__operators__.add_19/AddV2:z:0;custom_model_3/model_6/tf.expand_dims_6/ExpandDims:output:0*
T0*,
_output_shapes
:??????????2/
-custom_model_3/model_6/tf.math.multiply_6/Mul?
Acustom_model_3/model_6/tf.math.reduce_sum_6/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2C
Acustom_model_3/model_6/tf.math.reduce_sum_6/Sum/reduction_indices?
/custom_model_3/model_6/tf.math.reduce_sum_6/SumSum1custom_model_3/model_6/tf.math.multiply_6/Mul:z:0Jcustom_model_3/model_6/tf.math.reduce_sum_6/Sum/reduction_indices:output:0*
T0*(
_output_shapes
:??????????21
/custom_model_3/model_6/tf.math.reduce_sum_6/Sum?
&custom_model_3/model_7/flatten_7/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2(
&custom_model_3/model_7/flatten_7/Const?
(custom_model_3/model_7/flatten_7/ReshapeReshapecards1/custom_model_3/model_7/flatten_7/Const:output:0*
T0*'
_output_shapes
:?????????2*
(custom_model_3/model_7/flatten_7/Reshape?
Bcustom_model_3/model_7/tf.clip_by_value_10/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 * $tI2D
Bcustom_model_3/model_7/tf.clip_by_value_10/clip_by_value/Minimum/y?
@custom_model_3/model_7/tf.clip_by_value_10/clip_by_value/MinimumMinimum1custom_model_3/model_7/flatten_7/Reshape:output:0Kcustom_model_3/model_7/tf.clip_by_value_10/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2B
@custom_model_3/model_7/tf.clip_by_value_10/clip_by_value/Minimum?
:custom_model_3/model_7/tf.clip_by_value_10/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2<
:custom_model_3/model_7/tf.clip_by_value_10/clip_by_value/y?
8custom_model_3/model_7/tf.clip_by_value_10/clip_by_valueMaximumDcustom_model_3/model_7/tf.clip_by_value_10/clip_by_value/Minimum:z:0Ccustom_model_3/model_7/tf.clip_by_value_10/clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????2:
8custom_model_3/model_7/tf.clip_by_value_10/clip_by_value?
:custom_model_3/model_7/tf.compat.v1.floor_div_7/FloorDiv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2<
:custom_model_3/model_7/tf.compat.v1.floor_div_7/FloorDiv/y?
8custom_model_3/model_7/tf.compat.v1.floor_div_7/FloorDivFloorDiv<custom_model_3/model_7/tf.clip_by_value_10/clip_by_value:z:0Ccustom_model_3/model_7/tf.compat.v1.floor_div_7/FloorDiv/y:output:0*
T0*'
_output_shapes
:?????????2:
8custom_model_3/model_7/tf.compat.v1.floor_div_7/FloorDiv?
<custom_model_3/model_7/tf.math.greater_equal_10/GreaterEqualGreaterEqual1custom_model_3/model_7/flatten_7/Reshape:output:0>custom_model_3_model_7_tf_math_greater_equal_10_greaterequal_y*
T0*'
_output_shapes
:?????????2>
<custom_model_3/model_7/tf.math.greater_equal_10/GreaterEqual?
4custom_model_3/model_7/tf.math.floormod_7/FloorMod/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@26
4custom_model_3/model_7/tf.math.floormod_7/FloorMod/y?
2custom_model_3/model_7/tf.math.floormod_7/FloorModFloorMod<custom_model_3/model_7/tf.clip_by_value_10/clip_by_value:z:0=custom_model_3/model_7/tf.math.floormod_7/FloorMod/y:output:0*
T0*'
_output_shapes
:?????????24
2custom_model_3/model_7/tf.math.floormod_7/FloorMod?
(custom_model_3/model_7/embedding_23/CastCast<custom_model_3/model_7/tf.clip_by_value_10/clip_by_value:z:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2*
(custom_model_3/model_7/embedding_23/Cast?
4custom_model_3/model_7/embedding_23/embedding_lookupResourceGather=custom_model_3_model_7_embedding_23_embedding_lookup_20517259,custom_model_3/model_7/embedding_23/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*P
_classF
DBloc:@custom_model_3/model_7/embedding_23/embedding_lookup/20517259*,
_output_shapes
:??????????*
dtype026
4custom_model_3/model_7/embedding_23/embedding_lookup?
=custom_model_3/model_7/embedding_23/embedding_lookup/IdentityIdentity=custom_model_3/model_7/embedding_23/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*P
_classF
DBloc:@custom_model_3/model_7/embedding_23/embedding_lookup/20517259*,
_output_shapes
:??????????2?
=custom_model_3/model_7/embedding_23/embedding_lookup/Identity?
?custom_model_3/model_7/embedding_23/embedding_lookup/Identity_1IdentityFcustom_model_3/model_7/embedding_23/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????2A
?custom_model_3/model_7/embedding_23/embedding_lookup/Identity_1?
(custom_model_3/model_7/embedding_21/CastCast<custom_model_3/model_7/tf.compat.v1.floor_div_7/FloorDiv:z:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2*
(custom_model_3/model_7/embedding_21/Cast?
4custom_model_3/model_7/embedding_21/embedding_lookupResourceGather=custom_model_3_model_7_embedding_21_embedding_lookup_20517265,custom_model_3/model_7/embedding_21/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*P
_classF
DBloc:@custom_model_3/model_7/embedding_21/embedding_lookup/20517265*,
_output_shapes
:??????????*
dtype026
4custom_model_3/model_7/embedding_21/embedding_lookup?
=custom_model_3/model_7/embedding_21/embedding_lookup/IdentityIdentity=custom_model_3/model_7/embedding_21/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*P
_classF
DBloc:@custom_model_3/model_7/embedding_21/embedding_lookup/20517265*,
_output_shapes
:??????????2?
=custom_model_3/model_7/embedding_21/embedding_lookup/Identity?
?custom_model_3/model_7/embedding_21/embedding_lookup/Identity_1IdentityFcustom_model_3/model_7/embedding_21/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????2A
?custom_model_3/model_7/embedding_21/embedding_lookup/Identity_1?
&custom_model_3/model_7/tf.cast_10/CastCast@custom_model_3/model_7/tf.math.greater_equal_10/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2(
&custom_model_3/model_7/tf.cast_10/Cast?
4custom_model_3/model_7/tf.__operators__.add_20/AddV2AddV2Hcustom_model_3/model_7/embedding_23/embedding_lookup/Identity_1:output:0Hcustom_model_3/model_7/embedding_21/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:??????????26
4custom_model_3/model_7/tf.__operators__.add_20/AddV2?
(custom_model_3/model_7/embedding_22/CastCast6custom_model_3/model_7/tf.math.floormod_7/FloorMod:z:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2*
(custom_model_3/model_7/embedding_22/Cast?
4custom_model_3/model_7/embedding_22/embedding_lookupResourceGather=custom_model_3_model_7_embedding_22_embedding_lookup_20517273,custom_model_3/model_7/embedding_22/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*P
_classF
DBloc:@custom_model_3/model_7/embedding_22/embedding_lookup/20517273*,
_output_shapes
:??????????*
dtype026
4custom_model_3/model_7/embedding_22/embedding_lookup?
=custom_model_3/model_7/embedding_22/embedding_lookup/IdentityIdentity=custom_model_3/model_7/embedding_22/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*P
_classF
DBloc:@custom_model_3/model_7/embedding_22/embedding_lookup/20517273*,
_output_shapes
:??????????2?
=custom_model_3/model_7/embedding_22/embedding_lookup/Identity?
?custom_model_3/model_7/embedding_22/embedding_lookup/Identity_1IdentityFcustom_model_3/model_7/embedding_22/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????2A
?custom_model_3/model_7/embedding_22/embedding_lookup/Identity_1?
4custom_model_3/model_7/tf.__operators__.add_21/AddV2AddV28custom_model_3/model_7/tf.__operators__.add_20/AddV2:z:0Hcustom_model_3/model_7/embedding_22/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:??????????26
4custom_model_3/model_7/tf.__operators__.add_21/AddV2?
6custom_model_3/model_7/tf.expand_dims_7/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????28
6custom_model_3/model_7/tf.expand_dims_7/ExpandDims/dim?
2custom_model_3/model_7/tf.expand_dims_7/ExpandDims
ExpandDims*custom_model_3/model_7/tf.cast_10/Cast:y:0?custom_model_3/model_7/tf.expand_dims_7/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????24
2custom_model_3/model_7/tf.expand_dims_7/ExpandDims?
-custom_model_3/model_7/tf.math.multiply_7/MulMul8custom_model_3/model_7/tf.__operators__.add_21/AddV2:z:0;custom_model_3/model_7/tf.expand_dims_7/ExpandDims:output:0*
T0*,
_output_shapes
:??????????2/
-custom_model_3/model_7/tf.math.multiply_7/Mul?
Acustom_model_3/model_7/tf.math.reduce_sum_7/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2C
Acustom_model_3/model_7/tf.math.reduce_sum_7/Sum/reduction_indices?
/custom_model_3/model_7/tf.math.reduce_sum_7/SumSum1custom_model_3/model_7/tf.math.multiply_7/Mul:z:0Jcustom_model_3/model_7/tf.math.reduce_sum_7/Sum/reduction_indices:output:0*
T0*(
_output_shapes
:??????????21
/custom_model_3/model_7/tf.math.reduce_sum_7/Sum?
8custom_model_3/tf.clip_by_value_11/clip_by_value/MinimumMinimumbets:custom_model_3_tf_clip_by_value_11_clip_by_value_minimum_y*
T0*'
_output_shapes
:?????????
2:
8custom_model_3/tf.clip_by_value_11/clip_by_value/Minimum?
0custom_model_3/tf.clip_by_value_11/clip_by_valueMaximum<custom_model_3/tf.clip_by_value_11/clip_by_value/Minimum:z:02custom_model_3_tf_clip_by_value_11_clip_by_value_y*
T0*'
_output_shapes
:?????????
22
0custom_model_3/tf.clip_by_value_11/clip_by_value?
custom_model_3/tf.cast_11/CastCast8custom_model_3/tf.math.greater_equal_11/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????
2 
custom_model_3/tf.cast_11/Cast?
&custom_model_3/tf.concat_9/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2(
&custom_model_3/tf.concat_9/concat/axis?
!custom_model_3/tf.concat_9/concatConcatV28custom_model_3/model_6/tf.math.reduce_sum_6/Sum:output:08custom_model_3/model_7/tf.math.reduce_sum_7/Sum:output:0/custom_model_3/tf.concat_9/concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????2#
!custom_model_3/tf.concat_9/concat?
'custom_model_3/tf.concat_10/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2)
'custom_model_3/tf.concat_10/concat/axis?
"custom_model_3/tf.concat_10/concatConcatV24custom_model_3/tf.clip_by_value_11/clip_by_value:z:0"custom_model_3/tf.cast_11/Cast:y:00custom_model_3/tf.concat_10/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????2$
"custom_model_3/tf.concat_10/concat?
-custom_model_3/dense_27/MatMul/ReadVariableOpReadVariableOp6custom_model_3_dense_27_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02/
-custom_model_3/dense_27/MatMul/ReadVariableOp?
custom_model_3/dense_27/MatMulMatMul*custom_model_3/tf.concat_9/concat:output:05custom_model_3/dense_27/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2 
custom_model_3/dense_27/MatMul?
.custom_model_3/dense_27/BiasAdd/ReadVariableOpReadVariableOp7custom_model_3_dense_27_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype020
.custom_model_3/dense_27/BiasAdd/ReadVariableOp?
custom_model_3/dense_27/BiasAddBiasAdd(custom_model_3/dense_27/MatMul:product:06custom_model_3/dense_27/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2!
custom_model_3/dense_27/BiasAdd?
custom_model_3/dense_27/ReluRelu(custom_model_3/dense_27/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
custom_model_3/dense_27/Relu?
-custom_model_3/dense_30/MatMul/ReadVariableOpReadVariableOp6custom_model_3_dense_30_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02/
-custom_model_3/dense_30/MatMul/ReadVariableOp?
custom_model_3/dense_30/MatMulMatMul+custom_model_3/tf.concat_10/concat:output:05custom_model_3/dense_30/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2 
custom_model_3/dense_30/MatMul?
.custom_model_3/dense_30/BiasAdd/ReadVariableOpReadVariableOp7custom_model_3_dense_30_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype020
.custom_model_3/dense_30/BiasAdd/ReadVariableOp?
custom_model_3/dense_30/BiasAddBiasAdd(custom_model_3/dense_30/MatMul:product:06custom_model_3/dense_30/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2!
custom_model_3/dense_30/BiasAdd?
-custom_model_3/dense_28/MatMul/ReadVariableOpReadVariableOp6custom_model_3_dense_28_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02/
-custom_model_3/dense_28/MatMul/ReadVariableOp?
custom_model_3/dense_28/MatMulMatMul*custom_model_3/dense_27/Relu:activations:05custom_model_3/dense_28/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2 
custom_model_3/dense_28/MatMul?
.custom_model_3/dense_28/BiasAdd/ReadVariableOpReadVariableOp7custom_model_3_dense_28_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype020
.custom_model_3/dense_28/BiasAdd/ReadVariableOp?
custom_model_3/dense_28/BiasAddBiasAdd(custom_model_3/dense_28/MatMul:product:06custom_model_3/dense_28/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2!
custom_model_3/dense_28/BiasAdd?
custom_model_3/dense_28/ReluRelu(custom_model_3/dense_28/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
custom_model_3/dense_28/Relu?
-custom_model_3/dense_29/MatMul/ReadVariableOpReadVariableOp6custom_model_3_dense_29_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02/
-custom_model_3/dense_29/MatMul/ReadVariableOp?
custom_model_3/dense_29/MatMulMatMul*custom_model_3/dense_28/Relu:activations:05custom_model_3/dense_29/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2 
custom_model_3/dense_29/MatMul?
.custom_model_3/dense_29/BiasAdd/ReadVariableOpReadVariableOp7custom_model_3_dense_29_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype020
.custom_model_3/dense_29/BiasAdd/ReadVariableOp?
custom_model_3/dense_29/BiasAddBiasAdd(custom_model_3/dense_29/MatMul:product:06custom_model_3/dense_29/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2!
custom_model_3/dense_29/BiasAdd?
custom_model_3/dense_29/ReluRelu(custom_model_3/dense_29/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
custom_model_3/dense_29/Relu?
-custom_model_3/dense_31/MatMul/ReadVariableOpReadVariableOp6custom_model_3_dense_31_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02/
-custom_model_3/dense_31/MatMul/ReadVariableOp?
custom_model_3/dense_31/MatMulMatMul(custom_model_3/dense_30/BiasAdd:output:05custom_model_3/dense_31/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2 
custom_model_3/dense_31/MatMul?
.custom_model_3/dense_31/BiasAdd/ReadVariableOpReadVariableOp7custom_model_3_dense_31_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype020
.custom_model_3/dense_31/BiasAdd/ReadVariableOp?
custom_model_3/dense_31/BiasAddBiasAdd(custom_model_3/dense_31/MatMul:product:06custom_model_3/dense_31/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2!
custom_model_3/dense_31/BiasAdd?
'custom_model_3/tf.concat_11/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2)
'custom_model_3/tf.concat_11/concat/axis?
"custom_model_3/tf.concat_11/concatConcatV2*custom_model_3/dense_29/Relu:activations:0(custom_model_3/dense_31/BiasAdd:output:00custom_model_3/tf.concat_11/concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????2$
"custom_model_3/tf.concat_11/concat?
-custom_model_3/dense_32/MatMul/ReadVariableOpReadVariableOp6custom_model_3_dense_32_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02/
-custom_model_3/dense_32/MatMul/ReadVariableOp?
custom_model_3/dense_32/MatMulMatMul+custom_model_3/tf.concat_11/concat:output:05custom_model_3/dense_32/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2 
custom_model_3/dense_32/MatMul?
.custom_model_3/dense_32/BiasAdd/ReadVariableOpReadVariableOp7custom_model_3_dense_32_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype020
.custom_model_3/dense_32/BiasAdd/ReadVariableOp?
custom_model_3/dense_32/BiasAddBiasAdd(custom_model_3/dense_32/MatMul:product:06custom_model_3/dense_32/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2!
custom_model_3/dense_32/BiasAdd?
 custom_model_3/tf.nn.relu_9/ReluRelu(custom_model_3/dense_32/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2"
 custom_model_3/tf.nn.relu_9/Relu?
-custom_model_3/dense_33/MatMul/ReadVariableOpReadVariableOp6custom_model_3_dense_33_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02/
-custom_model_3/dense_33/MatMul/ReadVariableOp?
custom_model_3/dense_33/MatMulMatMul.custom_model_3/tf.nn.relu_9/Relu:activations:05custom_model_3/dense_33/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2 
custom_model_3/dense_33/MatMul?
.custom_model_3/dense_33/BiasAdd/ReadVariableOpReadVariableOp7custom_model_3_dense_33_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype020
.custom_model_3/dense_33/BiasAdd/ReadVariableOp?
custom_model_3/dense_33/BiasAddBiasAdd(custom_model_3/dense_33/MatMul:product:06custom_model_3/dense_33/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2!
custom_model_3/dense_33/BiasAdd?
,custom_model_3/tf.__operators__.add_22/AddV2AddV2(custom_model_3/dense_33/BiasAdd:output:0.custom_model_3/tf.nn.relu_9/Relu:activations:0*
T0*(
_output_shapes
:??????????2.
,custom_model_3/tf.__operators__.add_22/AddV2?
!custom_model_3/tf.nn.relu_10/ReluRelu0custom_model_3/tf.__operators__.add_22/AddV2:z:0*
T0*(
_output_shapes
:??????????2#
!custom_model_3/tf.nn.relu_10/Relu?
-custom_model_3/dense_34/MatMul/ReadVariableOpReadVariableOp6custom_model_3_dense_34_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02/
-custom_model_3/dense_34/MatMul/ReadVariableOp?
custom_model_3/dense_34/MatMulMatMul/custom_model_3/tf.nn.relu_10/Relu:activations:05custom_model_3/dense_34/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2 
custom_model_3/dense_34/MatMul?
.custom_model_3/dense_34/BiasAdd/ReadVariableOpReadVariableOp7custom_model_3_dense_34_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype020
.custom_model_3/dense_34/BiasAdd/ReadVariableOp?
custom_model_3/dense_34/BiasAddBiasAdd(custom_model_3/dense_34/MatMul:product:06custom_model_3/dense_34/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2!
custom_model_3/dense_34/BiasAdd?
,custom_model_3/tf.__operators__.add_23/AddV2AddV2(custom_model_3/dense_34/BiasAdd:output:0/custom_model_3/tf.nn.relu_10/Relu:activations:0*
T0*(
_output_shapes
:??????????2.
,custom_model_3/tf.__operators__.add_23/AddV2?
!custom_model_3/tf.nn.relu_11/ReluRelu0custom_model_3/tf.__operators__.add_23/AddV2:z:0*
T0*(
_output_shapes
:??????????2#
!custom_model_3/tf.nn.relu_11/Relu?
Acustom_model_3/normalize_3/normalization_3/Reshape/ReadVariableOpReadVariableOpJcustom_model_3_normalize_3_normalization_3_reshape_readvariableop_resource*
_output_shapes	
:?*
dtype02C
Acustom_model_3/normalize_3/normalization_3/Reshape/ReadVariableOp?
8custom_model_3/normalize_3/normalization_3/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2:
8custom_model_3/normalize_3/normalization_3/Reshape/shape?
2custom_model_3/normalize_3/normalization_3/ReshapeReshapeIcustom_model_3/normalize_3/normalization_3/Reshape/ReadVariableOp:value:0Acustom_model_3/normalize_3/normalization_3/Reshape/shape:output:0*
T0*
_output_shapes
:	?24
2custom_model_3/normalize_3/normalization_3/Reshape?
Ccustom_model_3/normalize_3/normalization_3/Reshape_1/ReadVariableOpReadVariableOpLcustom_model_3_normalize_3_normalization_3_reshape_1_readvariableop_resource*
_output_shapes	
:?*
dtype02E
Ccustom_model_3/normalize_3/normalization_3/Reshape_1/ReadVariableOp?
:custom_model_3/normalize_3/normalization_3/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2<
:custom_model_3/normalize_3/normalization_3/Reshape_1/shape?
4custom_model_3/normalize_3/normalization_3/Reshape_1ReshapeKcustom_model_3/normalize_3/normalization_3/Reshape_1/ReadVariableOp:value:0Ccustom_model_3/normalize_3/normalization_3/Reshape_1/shape:output:0*
T0*
_output_shapes
:	?26
4custom_model_3/normalize_3/normalization_3/Reshape_1?
.custom_model_3/normalize_3/normalization_3/subSub/custom_model_3/tf.nn.relu_11/Relu:activations:0;custom_model_3/normalize_3/normalization_3/Reshape:output:0*
T0*(
_output_shapes
:??????????20
.custom_model_3/normalize_3/normalization_3/sub?
/custom_model_3/normalize_3/normalization_3/SqrtSqrt=custom_model_3/normalize_3/normalization_3/Reshape_1:output:0*
T0*
_output_shapes
:	?21
/custom_model_3/normalize_3/normalization_3/Sqrt?
4custom_model_3/normalize_3/normalization_3/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???326
4custom_model_3/normalize_3/normalization_3/Maximum/y?
2custom_model_3/normalize_3/normalization_3/MaximumMaximum3custom_model_3/normalize_3/normalization_3/Sqrt:y:0=custom_model_3/normalize_3/normalization_3/Maximum/y:output:0*
T0*
_output_shapes
:	?24
2custom_model_3/normalize_3/normalization_3/Maximum?
2custom_model_3/normalize_3/normalization_3/truedivRealDiv2custom_model_3/normalize_3/normalization_3/sub:z:06custom_model_3/normalize_3/normalization_3/Maximum:z:0*
T0*(
_output_shapes
:??????????24
2custom_model_3/normalize_3/normalization_3/truediv?
-custom_model_3/dense_35/MatMul/ReadVariableOpReadVariableOp6custom_model_3_dense_35_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02/
-custom_model_3/dense_35/MatMul/ReadVariableOp?
custom_model_3/dense_35/MatMulMatMul6custom_model_3/normalize_3/normalization_3/truediv:z:05custom_model_3/dense_35/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2 
custom_model_3/dense_35/MatMul?
.custom_model_3/dense_35/BiasAdd/ReadVariableOpReadVariableOp7custom_model_3_dense_35_biasadd_readvariableop_resource*
_output_shapes
:*
dtype020
.custom_model_3/dense_35/BiasAdd/ReadVariableOp?
custom_model_3/dense_35/BiasAddBiasAdd(custom_model_3/dense_35/MatMul:product:06custom_model_3/dense_35/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2!
custom_model_3/dense_35/BiasAdd?
IdentityIdentity(custom_model_3/dense_35/BiasAdd:output:0/^custom_model_3/dense_27/BiasAdd/ReadVariableOp.^custom_model_3/dense_27/MatMul/ReadVariableOp/^custom_model_3/dense_28/BiasAdd/ReadVariableOp.^custom_model_3/dense_28/MatMul/ReadVariableOp/^custom_model_3/dense_29/BiasAdd/ReadVariableOp.^custom_model_3/dense_29/MatMul/ReadVariableOp/^custom_model_3/dense_30/BiasAdd/ReadVariableOp.^custom_model_3/dense_30/MatMul/ReadVariableOp/^custom_model_3/dense_31/BiasAdd/ReadVariableOp.^custom_model_3/dense_31/MatMul/ReadVariableOp/^custom_model_3/dense_32/BiasAdd/ReadVariableOp.^custom_model_3/dense_32/MatMul/ReadVariableOp/^custom_model_3/dense_33/BiasAdd/ReadVariableOp.^custom_model_3/dense_33/MatMul/ReadVariableOp/^custom_model_3/dense_34/BiasAdd/ReadVariableOp.^custom_model_3/dense_34/MatMul/ReadVariableOp/^custom_model_3/dense_35/BiasAdd/ReadVariableOp.^custom_model_3/dense_35/MatMul/ReadVariableOp5^custom_model_3/model_6/embedding_18/embedding_lookup5^custom_model_3/model_6/embedding_19/embedding_lookup5^custom_model_3/model_6/embedding_20/embedding_lookup5^custom_model_3/model_7/embedding_21/embedding_lookup5^custom_model_3/model_7/embedding_22/embedding_lookup5^custom_model_3/model_7/embedding_23/embedding_lookupB^custom_model_3/normalize_3/normalization_3/Reshape/ReadVariableOpD^custom_model_3/normalize_3/normalization_3/Reshape_1/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????:?????????:?????????
: : :::: :::: : ::::::::::::::::::::2`
.custom_model_3/dense_27/BiasAdd/ReadVariableOp.custom_model_3/dense_27/BiasAdd/ReadVariableOp2^
-custom_model_3/dense_27/MatMul/ReadVariableOp-custom_model_3/dense_27/MatMul/ReadVariableOp2`
.custom_model_3/dense_28/BiasAdd/ReadVariableOp.custom_model_3/dense_28/BiasAdd/ReadVariableOp2^
-custom_model_3/dense_28/MatMul/ReadVariableOp-custom_model_3/dense_28/MatMul/ReadVariableOp2`
.custom_model_3/dense_29/BiasAdd/ReadVariableOp.custom_model_3/dense_29/BiasAdd/ReadVariableOp2^
-custom_model_3/dense_29/MatMul/ReadVariableOp-custom_model_3/dense_29/MatMul/ReadVariableOp2`
.custom_model_3/dense_30/BiasAdd/ReadVariableOp.custom_model_3/dense_30/BiasAdd/ReadVariableOp2^
-custom_model_3/dense_30/MatMul/ReadVariableOp-custom_model_3/dense_30/MatMul/ReadVariableOp2`
.custom_model_3/dense_31/BiasAdd/ReadVariableOp.custom_model_3/dense_31/BiasAdd/ReadVariableOp2^
-custom_model_3/dense_31/MatMul/ReadVariableOp-custom_model_3/dense_31/MatMul/ReadVariableOp2`
.custom_model_3/dense_32/BiasAdd/ReadVariableOp.custom_model_3/dense_32/BiasAdd/ReadVariableOp2^
-custom_model_3/dense_32/MatMul/ReadVariableOp-custom_model_3/dense_32/MatMul/ReadVariableOp2`
.custom_model_3/dense_33/BiasAdd/ReadVariableOp.custom_model_3/dense_33/BiasAdd/ReadVariableOp2^
-custom_model_3/dense_33/MatMul/ReadVariableOp-custom_model_3/dense_33/MatMul/ReadVariableOp2`
.custom_model_3/dense_34/BiasAdd/ReadVariableOp.custom_model_3/dense_34/BiasAdd/ReadVariableOp2^
-custom_model_3/dense_34/MatMul/ReadVariableOp-custom_model_3/dense_34/MatMul/ReadVariableOp2`
.custom_model_3/dense_35/BiasAdd/ReadVariableOp.custom_model_3/dense_35/BiasAdd/ReadVariableOp2^
-custom_model_3/dense_35/MatMul/ReadVariableOp-custom_model_3/dense_35/MatMul/ReadVariableOp2l
4custom_model_3/model_6/embedding_18/embedding_lookup4custom_model_3/model_6/embedding_18/embedding_lookup2l
4custom_model_3/model_6/embedding_19/embedding_lookup4custom_model_3/model_6/embedding_19/embedding_lookup2l
4custom_model_3/model_6/embedding_20/embedding_lookup4custom_model_3/model_6/embedding_20/embedding_lookup2l
4custom_model_3/model_7/embedding_21/embedding_lookup4custom_model_3/model_7/embedding_21/embedding_lookup2l
4custom_model_3/model_7/embedding_22/embedding_lookup4custom_model_3/model_7/embedding_22/embedding_lookup2l
4custom_model_3/model_7/embedding_23/embedding_lookup4custom_model_3/model_7/embedding_23/embedding_lookup2?
Acustom_model_3/normalize_3/normalization_3/Reshape/ReadVariableOpAcustom_model_3/normalize_3/normalization_3/Reshape/ReadVariableOp2?
Ccustom_model_3/normalize_3/normalization_3/Reshape_1/ReadVariableOpCcustom_model_3/normalize_3/normalization_3/Reshape_1/ReadVariableOp:O K
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
?
?
1__inference_custom_model_3_layer_call_fn_20518603

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
L__inference_custom_model_3_layer_call_and_return_conditional_losses_205185382
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
?	
?
F__inference_dense_27_layer_call_and_return_conditional_losses_20519391

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
?8
?
E__inference_model_6_layer_call_and_return_conditional_losses_20519202

inputs*
&tf_math_greater_equal_9_greaterequal_y*
&embedding_20_embedding_lookup_20519176*
&embedding_18_embedding_lookup_20519182*
&embedding_19_embedding_lookup_20519190
identity??embedding_18/embedding_lookup?embedding_19/embedding_lookup?embedding_20/embedding_lookups
flatten_6/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
flatten_6/Const?
flatten_6/ReshapeReshapeinputsflatten_6/Const:output:0*
T0*'
_output_shapes
:?????????2
flatten_6/Reshape?
*tf.clip_by_value_9/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 * $tI2,
*tf.clip_by_value_9/clip_by_value/Minimum/y?
(tf.clip_by_value_9/clip_by_value/MinimumMinimumflatten_6/Reshape:output:03tf.clip_by_value_9/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2*
(tf.clip_by_value_9/clip_by_value/Minimum?
"tf.clip_by_value_9/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"tf.clip_by_value_9/clip_by_value/y?
 tf.clip_by_value_9/clip_by_valueMaximum,tf.clip_by_value_9/clip_by_value/Minimum:z:0+tf.clip_by_value_9/clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????2"
 tf.clip_by_value_9/clip_by_value?
#tf.compat.v1.floor_div_6/FloorDiv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2%
#tf.compat.v1.floor_div_6/FloorDiv/y?
!tf.compat.v1.floor_div_6/FloorDivFloorDiv$tf.clip_by_value_9/clip_by_value:z:0,tf.compat.v1.floor_div_6/FloorDiv/y:output:0*
T0*'
_output_shapes
:?????????2#
!tf.compat.v1.floor_div_6/FloorDiv?
$tf.math.greater_equal_9/GreaterEqualGreaterEqualflatten_6/Reshape:output:0&tf_math_greater_equal_9_greaterequal_y*
T0*'
_output_shapes
:?????????2&
$tf.math.greater_equal_9/GreaterEqual?
tf.math.floormod_6/FloorMod/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2
tf.math.floormod_6/FloorMod/y?
tf.math.floormod_6/FloorModFloorMod$tf.clip_by_value_9/clip_by_value:z:0&tf.math.floormod_6/FloorMod/y:output:0*
T0*'
_output_shapes
:?????????2
tf.math.floormod_6/FloorMod?
embedding_20/CastCast$tf.clip_by_value_9/clip_by_value:z:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2
embedding_20/Cast?
embedding_20/embedding_lookupResourceGather&embedding_20_embedding_lookup_20519176embedding_20/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*9
_class/
-+loc:@embedding_20/embedding_lookup/20519176*,
_output_shapes
:??????????*
dtype02
embedding_20/embedding_lookup?
&embedding_20/embedding_lookup/IdentityIdentity&embedding_20/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*9
_class/
-+loc:@embedding_20/embedding_lookup/20519176*,
_output_shapes
:??????????2(
&embedding_20/embedding_lookup/Identity?
(embedding_20/embedding_lookup/Identity_1Identity/embedding_20/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????2*
(embedding_20/embedding_lookup/Identity_1?
embedding_18/CastCast%tf.compat.v1.floor_div_6/FloorDiv:z:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2
embedding_18/Cast?
embedding_18/embedding_lookupResourceGather&embedding_18_embedding_lookup_20519182embedding_18/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*9
_class/
-+loc:@embedding_18/embedding_lookup/20519182*,
_output_shapes
:??????????*
dtype02
embedding_18/embedding_lookup?
&embedding_18/embedding_lookup/IdentityIdentity&embedding_18/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*9
_class/
-+loc:@embedding_18/embedding_lookup/20519182*,
_output_shapes
:??????????2(
&embedding_18/embedding_lookup/Identity?
(embedding_18/embedding_lookup/Identity_1Identity/embedding_18/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????2*
(embedding_18/embedding_lookup/Identity_1?
tf.cast_9/CastCast(tf.math.greater_equal_9/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2
tf.cast_9/Cast?
tf.__operators__.add_18/AddV2AddV21embedding_20/embedding_lookup/Identity_1:output:01embedding_18/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:??????????2
tf.__operators__.add_18/AddV2?
embedding_19/CastCasttf.math.floormod_6/FloorMod:z:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2
embedding_19/Cast?
embedding_19/embedding_lookupResourceGather&embedding_19_embedding_lookup_20519190embedding_19/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*9
_class/
-+loc:@embedding_19/embedding_lookup/20519190*,
_output_shapes
:??????????*
dtype02
embedding_19/embedding_lookup?
&embedding_19/embedding_lookup/IdentityIdentity&embedding_19/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*9
_class/
-+loc:@embedding_19/embedding_lookup/20519190*,
_output_shapes
:??????????2(
&embedding_19/embedding_lookup/Identity?
(embedding_19/embedding_lookup/Identity_1Identity/embedding_19/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????2*
(embedding_19/embedding_lookup/Identity_1?
tf.__operators__.add_19/AddV2AddV2!tf.__operators__.add_18/AddV2:z:01embedding_19/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:??????????2
tf.__operators__.add_19/AddV2?
tf.expand_dims_6/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2!
tf.expand_dims_6/ExpandDims/dim?
tf.expand_dims_6/ExpandDims
ExpandDimstf.cast_9/Cast:y:0(tf.expand_dims_6/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????2
tf.expand_dims_6/ExpandDims?
tf.math.multiply_6/MulMul!tf.__operators__.add_19/AddV2:z:0$tf.expand_dims_6/ExpandDims:output:0*
T0*,
_output_shapes
:??????????2
tf.math.multiply_6/Mul?
*tf.math.reduce_sum_6/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2,
*tf.math.reduce_sum_6/Sum/reduction_indices?
tf.math.reduce_sum_6/SumSumtf.math.multiply_6/Mul:z:03tf.math.reduce_sum_6/Sum/reduction_indices:output:0*
T0*(
_output_shapes
:??????????2
tf.math.reduce_sum_6/Sum?
IdentityIdentity!tf.math.reduce_sum_6/Sum:output:0^embedding_18/embedding_lookup^embedding_19/embedding_lookup^embedding_20/embedding_lookup*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:?????????: :::2>
embedding_18/embedding_lookupembedding_18/embedding_lookup2>
embedding_19/embedding_lookupembedding_19/embedding_lookup2>
embedding_20/embedding_lookupembedding_20/embedding_lookup:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: 
?
?
*__inference_model_7_layer_call_fn_20519367

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
GPU2 *0J 8? *N
fIRG
E__inference_model_7_layer_call_and_return_conditional_losses_205177672
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
?
u
/__inference_embedding_23_layer_call_fn_20519670

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
J__inference_embedding_23_layer_call_and_return_conditional_losses_205176352
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
?
c
G__inference_flatten_7_layer_call_and_return_conditional_losses_20517607

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
F__inference_dense_35_layer_call_and_return_conditional_losses_20518171

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
.__inference_normalize_3_layer_call_fn_20519561
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
I__inference_normalize_3_layer_call_and_return_conditional_losses_205181452
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
?
?
1__inference_custom_model_3_layer_call_fn_20518442

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
L__inference_custom_model_3_layer_call_and_return_conditional_losses_205183772
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
?	
?
F__inference_dense_30_layer_call_and_return_conditional_losses_20519430

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
?-
?
E__inference_model_7_layer_call_and_return_conditional_losses_20517732
input_8+
'tf_math_greater_equal_10_greaterequal_y
embedding_23_20517714
embedding_21_20517717
embedding_22_20517722
identity??$embedding_21/StatefulPartitionedCall?$embedding_22/StatefulPartitionedCall?$embedding_23/StatefulPartitionedCall?
flatten_7/PartitionedCallPartitionedCallinput_8*
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
GPU2 *0J 8? *P
fKRI
G__inference_flatten_7_layer_call_and_return_conditional_losses_205176072
flatten_7/PartitionedCall?
+tf.clip_by_value_10/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 * $tI2-
+tf.clip_by_value_10/clip_by_value/Minimum/y?
)tf.clip_by_value_10/clip_by_value/MinimumMinimum"flatten_7/PartitionedCall:output:04tf.clip_by_value_10/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2+
)tf.clip_by_value_10/clip_by_value/Minimum?
#tf.clip_by_value_10/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2%
#tf.clip_by_value_10/clip_by_value/y?
!tf.clip_by_value_10/clip_by_valueMaximum-tf.clip_by_value_10/clip_by_value/Minimum:z:0,tf.clip_by_value_10/clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????2#
!tf.clip_by_value_10/clip_by_value?
#tf.compat.v1.floor_div_7/FloorDiv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2%
#tf.compat.v1.floor_div_7/FloorDiv/y?
!tf.compat.v1.floor_div_7/FloorDivFloorDiv%tf.clip_by_value_10/clip_by_value:z:0,tf.compat.v1.floor_div_7/FloorDiv/y:output:0*
T0*'
_output_shapes
:?????????2#
!tf.compat.v1.floor_div_7/FloorDiv?
%tf.math.greater_equal_10/GreaterEqualGreaterEqual"flatten_7/PartitionedCall:output:0'tf_math_greater_equal_10_greaterequal_y*
T0*'
_output_shapes
:?????????2'
%tf.math.greater_equal_10/GreaterEqual?
tf.math.floormod_7/FloorMod/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2
tf.math.floormod_7/FloorMod/y?
tf.math.floormod_7/FloorModFloorMod%tf.clip_by_value_10/clip_by_value:z:0&tf.math.floormod_7/FloorMod/y:output:0*
T0*'
_output_shapes
:?????????2
tf.math.floormod_7/FloorMod?
$embedding_23/StatefulPartitionedCallStatefulPartitionedCall%tf.clip_by_value_10/clip_by_value:z:0embedding_23_20517714*
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
J__inference_embedding_23_layer_call_and_return_conditional_losses_205176352&
$embedding_23/StatefulPartitionedCall?
$embedding_21/StatefulPartitionedCallStatefulPartitionedCall%tf.compat.v1.floor_div_7/FloorDiv:z:0embedding_21_20517717*
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
J__inference_embedding_21_layer_call_and_return_conditional_losses_205176572&
$embedding_21/StatefulPartitionedCall?
tf.cast_10/CastCast)tf.math.greater_equal_10/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2
tf.cast_10/Cast?
tf.__operators__.add_20/AddV2AddV2-embedding_23/StatefulPartitionedCall:output:0-embedding_21/StatefulPartitionedCall:output:0*
T0*,
_output_shapes
:??????????2
tf.__operators__.add_20/AddV2?
$embedding_22/StatefulPartitionedCallStatefulPartitionedCalltf.math.floormod_7/FloorMod:z:0embedding_22_20517722*
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
J__inference_embedding_22_layer_call_and_return_conditional_losses_205176812&
$embedding_22/StatefulPartitionedCall?
tf.__operators__.add_21/AddV2AddV2!tf.__operators__.add_20/AddV2:z:0-embedding_22/StatefulPartitionedCall:output:0*
T0*,
_output_shapes
:??????????2
tf.__operators__.add_21/AddV2?
tf.expand_dims_7/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2!
tf.expand_dims_7/ExpandDims/dim?
tf.expand_dims_7/ExpandDims
ExpandDimstf.cast_10/Cast:y:0(tf.expand_dims_7/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????2
tf.expand_dims_7/ExpandDims?
tf.math.multiply_7/MulMul!tf.__operators__.add_21/AddV2:z:0$tf.expand_dims_7/ExpandDims:output:0*
T0*,
_output_shapes
:??????????2
tf.math.multiply_7/Mul?
*tf.math.reduce_sum_7/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2,
*tf.math.reduce_sum_7/Sum/reduction_indices?
tf.math.reduce_sum_7/SumSumtf.math.multiply_7/Mul:z:03tf.math.reduce_sum_7/Sum/reduction_indices:output:0*
T0*(
_output_shapes
:??????????2
tf.math.reduce_sum_7/Sum?
IdentityIdentity!tf.math.reduce_sum_7/Sum:output:0%^embedding_21/StatefulPartitionedCall%^embedding_22/StatefulPartitionedCall%^embedding_23/StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:?????????: :::2L
$embedding_21/StatefulPartitionedCall$embedding_21/StatefulPartitionedCall2L
$embedding_22/StatefulPartitionedCall$embedding_22/StatefulPartitionedCall2L
$embedding_23/StatefulPartitionedCall$embedding_23/StatefulPartitionedCall:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_8:

_output_shapes
: 
?
?
+__inference_dense_31_layer_call_fn_20519478

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
F__inference_dense_31_layer_call_and_return_conditional_losses_205180272
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
?-
?
E__inference_model_6_layer_call_and_return_conditional_losses_20517541

inputs*
&tf_math_greater_equal_9_greaterequal_y
embedding_20_20517523
embedding_18_20517526
embedding_19_20517531
identity??$embedding_18/StatefulPartitionedCall?$embedding_19/StatefulPartitionedCall?$embedding_20/StatefulPartitionedCall?
flatten_6/PartitionedCallPartitionedCallinputs*
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
GPU2 *0J 8? *P
fKRI
G__inference_flatten_6_layer_call_and_return_conditional_losses_205173812
flatten_6/PartitionedCall?
*tf.clip_by_value_9/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 * $tI2,
*tf.clip_by_value_9/clip_by_value/Minimum/y?
(tf.clip_by_value_9/clip_by_value/MinimumMinimum"flatten_6/PartitionedCall:output:03tf.clip_by_value_9/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2*
(tf.clip_by_value_9/clip_by_value/Minimum?
"tf.clip_by_value_9/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"tf.clip_by_value_9/clip_by_value/y?
 tf.clip_by_value_9/clip_by_valueMaximum,tf.clip_by_value_9/clip_by_value/Minimum:z:0+tf.clip_by_value_9/clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????2"
 tf.clip_by_value_9/clip_by_value?
#tf.compat.v1.floor_div_6/FloorDiv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2%
#tf.compat.v1.floor_div_6/FloorDiv/y?
!tf.compat.v1.floor_div_6/FloorDivFloorDiv$tf.clip_by_value_9/clip_by_value:z:0,tf.compat.v1.floor_div_6/FloorDiv/y:output:0*
T0*'
_output_shapes
:?????????2#
!tf.compat.v1.floor_div_6/FloorDiv?
$tf.math.greater_equal_9/GreaterEqualGreaterEqual"flatten_6/PartitionedCall:output:0&tf_math_greater_equal_9_greaterequal_y*
T0*'
_output_shapes
:?????????2&
$tf.math.greater_equal_9/GreaterEqual?
tf.math.floormod_6/FloorMod/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2
tf.math.floormod_6/FloorMod/y?
tf.math.floormod_6/FloorModFloorMod$tf.clip_by_value_9/clip_by_value:z:0&tf.math.floormod_6/FloorMod/y:output:0*
T0*'
_output_shapes
:?????????2
tf.math.floormod_6/FloorMod?
$embedding_20/StatefulPartitionedCallStatefulPartitionedCall$tf.clip_by_value_9/clip_by_value:z:0embedding_20_20517523*
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
J__inference_embedding_20_layer_call_and_return_conditional_losses_205174092&
$embedding_20/StatefulPartitionedCall?
$embedding_18/StatefulPartitionedCallStatefulPartitionedCall%tf.compat.v1.floor_div_6/FloorDiv:z:0embedding_18_20517526*
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
J__inference_embedding_18_layer_call_and_return_conditional_losses_205174312&
$embedding_18/StatefulPartitionedCall?
tf.cast_9/CastCast(tf.math.greater_equal_9/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2
tf.cast_9/Cast?
tf.__operators__.add_18/AddV2AddV2-embedding_20/StatefulPartitionedCall:output:0-embedding_18/StatefulPartitionedCall:output:0*
T0*,
_output_shapes
:??????????2
tf.__operators__.add_18/AddV2?
$embedding_19/StatefulPartitionedCallStatefulPartitionedCalltf.math.floormod_6/FloorMod:z:0embedding_19_20517531*
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
J__inference_embedding_19_layer_call_and_return_conditional_losses_205174552&
$embedding_19/StatefulPartitionedCall?
tf.__operators__.add_19/AddV2AddV2!tf.__operators__.add_18/AddV2:z:0-embedding_19/StatefulPartitionedCall:output:0*
T0*,
_output_shapes
:??????????2
tf.__operators__.add_19/AddV2?
tf.expand_dims_6/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2!
tf.expand_dims_6/ExpandDims/dim?
tf.expand_dims_6/ExpandDims
ExpandDimstf.cast_9/Cast:y:0(tf.expand_dims_6/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????2
tf.expand_dims_6/ExpandDims?
tf.math.multiply_6/MulMul!tf.__operators__.add_19/AddV2:z:0$tf.expand_dims_6/ExpandDims:output:0*
T0*,
_output_shapes
:??????????2
tf.math.multiply_6/Mul?
*tf.math.reduce_sum_6/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2,
*tf.math.reduce_sum_6/Sum/reduction_indices?
tf.math.reduce_sum_6/SumSumtf.math.multiply_6/Mul:z:03tf.math.reduce_sum_6/Sum/reduction_indices:output:0*
T0*(
_output_shapes
:??????????2
tf.math.reduce_sum_6/Sum?
IdentityIdentity!tf.math.reduce_sum_6/Sum:output:0%^embedding_18/StatefulPartitionedCall%^embedding_19/StatefulPartitionedCall%^embedding_20/StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:?????????: :::2L
$embedding_18/StatefulPartitionedCall$embedding_18/StatefulPartitionedCall2L
$embedding_19/StatefulPartitionedCall$embedding_19/StatefulPartitionedCall2L
$embedding_20/StatefulPartitionedCall$embedding_20/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: 
?	
?
J__inference_embedding_21_layer_call_and_return_conditional_losses_20517657

inputs
embedding_lookup_20517651
identity??embedding_lookup]
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:?????????2
Cast?
embedding_lookupResourceGatherembedding_lookup_20517651Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*,
_class"
 loc:@embedding_lookup/20517651*,
_output_shapes
:??????????*
dtype02
embedding_lookup?
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*,
_class"
 loc:@embedding_lookup/20517651*,
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
?
H
,__inference_flatten_6_layer_call_fn_20519591

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
GPU2 *0J 8? *P
fKRI
G__inference_flatten_6_layer_call_and_return_conditional_losses_205173812
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
?
c
G__inference_flatten_6_layer_call_and_return_conditional_losses_20517381

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
?X
?

L__inference_custom_model_3_layer_call_and_return_conditional_losses_20518280

cards0

cards1
bets+
'tf_math_greater_equal_11_greaterequal_y
model_6_20518195
model_6_20518197
model_6_20518199
model_6_20518201
model_7_20518204
model_7_20518206
model_7_20518208
model_7_20518210/
+tf_clip_by_value_11_clip_by_value_minimum_y'
#tf_clip_by_value_11_clip_by_value_y
dense_27_20518222
dense_27_20518224
dense_30_20518227
dense_30_20518229
dense_28_20518232
dense_28_20518234
dense_29_20518237
dense_29_20518239
dense_31_20518242
dense_31_20518244
dense_32_20518249
dense_32_20518251
dense_33_20518255
dense_33_20518257
dense_34_20518262
dense_34_20518264
normalize_3_20518269
normalize_3_20518271
dense_35_20518274
dense_35_20518276
identity?? dense_27/StatefulPartitionedCall? dense_28/StatefulPartitionedCall? dense_29/StatefulPartitionedCall? dense_30/StatefulPartitionedCall? dense_31/StatefulPartitionedCall? dense_32/StatefulPartitionedCall? dense_33/StatefulPartitionedCall? dense_34/StatefulPartitionedCall? dense_35/StatefulPartitionedCall?model_6/StatefulPartitionedCall?model_7/StatefulPartitionedCall?#normalize_3/StatefulPartitionedCall?
%tf.math.greater_equal_11/GreaterEqualGreaterEqualbets'tf_math_greater_equal_11_greaterequal_y*
T0*'
_output_shapes
:?????????
2'
%tf.math.greater_equal_11/GreaterEqual?
model_6/StatefulPartitionedCallStatefulPartitionedCallcards0model_6_20518195model_6_20518197model_6_20518199model_6_20518201*
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
GPU2 *0J 8? *N
fIRG
E__inference_model_6_layer_call_and_return_conditional_losses_205175862!
model_6/StatefulPartitionedCall?
model_7/StatefulPartitionedCallStatefulPartitionedCallcards1model_7_20518204model_7_20518206model_7_20518208model_7_20518210*
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
GPU2 *0J 8? *N
fIRG
E__inference_model_7_layer_call_and_return_conditional_losses_205178122!
model_7/StatefulPartitionedCall?
)tf.clip_by_value_11/clip_by_value/MinimumMinimumbets+tf_clip_by_value_11_clip_by_value_minimum_y*
T0*'
_output_shapes
:?????????
2+
)tf.clip_by_value_11/clip_by_value/Minimum?
!tf.clip_by_value_11/clip_by_valueMaximum-tf.clip_by_value_11/clip_by_value/Minimum:z:0#tf_clip_by_value_11_clip_by_value_y*
T0*'
_output_shapes
:?????????
2#
!tf.clip_by_value_11/clip_by_value?
tf.cast_11/CastCast)tf.math.greater_equal_11/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????
2
tf.cast_11/Castt
tf.concat_9/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
tf.concat_9/concat/axis?
tf.concat_9/concatConcatV2(model_6/StatefulPartitionedCall:output:0(model_7/StatefulPartitionedCall:output:0 tf.concat_9/concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????2
tf.concat_9/concat
tf.concat_10/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
tf.concat_10/concat/axis?
tf.concat_10/concatConcatV2%tf.clip_by_value_11/clip_by_value:z:0tf.cast_11/Cast:y:0!tf.concat_10/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????2
tf.concat_10/concat?
 dense_27/StatefulPartitionedCallStatefulPartitionedCalltf.concat_9/concat:output:0dense_27_20518222dense_27_20518224*
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
F__inference_dense_27_layer_call_and_return_conditional_losses_205179212"
 dense_27/StatefulPartitionedCall?
 dense_30/StatefulPartitionedCallStatefulPartitionedCalltf.concat_10/concat:output:0dense_30_20518227dense_30_20518229*
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
F__inference_dense_30_layer_call_and_return_conditional_losses_205179472"
 dense_30/StatefulPartitionedCall?
 dense_28/StatefulPartitionedCallStatefulPartitionedCall)dense_27/StatefulPartitionedCall:output:0dense_28_20518232dense_28_20518234*
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
F__inference_dense_28_layer_call_and_return_conditional_losses_205179742"
 dense_28/StatefulPartitionedCall?
 dense_29/StatefulPartitionedCallStatefulPartitionedCall)dense_28/StatefulPartitionedCall:output:0dense_29_20518237dense_29_20518239*
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
F__inference_dense_29_layer_call_and_return_conditional_losses_205180012"
 dense_29/StatefulPartitionedCall?
 dense_31/StatefulPartitionedCallStatefulPartitionedCall)dense_30/StatefulPartitionedCall:output:0dense_31_20518242dense_31_20518244*
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
F__inference_dense_31_layer_call_and_return_conditional_losses_205180272"
 dense_31/StatefulPartitionedCall
tf.concat_11/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
tf.concat_11/concat/axis?
tf.concat_11/concatConcatV2)dense_29/StatefulPartitionedCall:output:0)dense_31/StatefulPartitionedCall:output:0!tf.concat_11/concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????2
tf.concat_11/concat?
 dense_32/StatefulPartitionedCallStatefulPartitionedCalltf.concat_11/concat:output:0dense_32_20518249dense_32_20518251*
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
F__inference_dense_32_layer_call_and_return_conditional_losses_205180552"
 dense_32/StatefulPartitionedCall?
tf.nn.relu_9/ReluRelu)dense_32/StatefulPartitionedCall:output:0*
T0*(
_output_shapes
:??????????2
tf.nn.relu_9/Relu?
 dense_33/StatefulPartitionedCallStatefulPartitionedCalltf.nn.relu_9/Relu:activations:0dense_33_20518255dense_33_20518257*
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
F__inference_dense_33_layer_call_and_return_conditional_losses_205180822"
 dense_33/StatefulPartitionedCall?
tf.__operators__.add_22/AddV2AddV2)dense_33/StatefulPartitionedCall:output:0tf.nn.relu_9/Relu:activations:0*
T0*(
_output_shapes
:??????????2
tf.__operators__.add_22/AddV2?
tf.nn.relu_10/ReluRelu!tf.__operators__.add_22/AddV2:z:0*
T0*(
_output_shapes
:??????????2
tf.nn.relu_10/Relu?
 dense_34/StatefulPartitionedCallStatefulPartitionedCall tf.nn.relu_10/Relu:activations:0dense_34_20518262dense_34_20518264*
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
F__inference_dense_34_layer_call_and_return_conditional_losses_205181102"
 dense_34/StatefulPartitionedCall?
tf.__operators__.add_23/AddV2AddV2)dense_34/StatefulPartitionedCall:output:0 tf.nn.relu_10/Relu:activations:0*
T0*(
_output_shapes
:??????????2
tf.__operators__.add_23/AddV2?
tf.nn.relu_11/ReluRelu!tf.__operators__.add_23/AddV2:z:0*
T0*(
_output_shapes
:??????????2
tf.nn.relu_11/Relu?
#normalize_3/StatefulPartitionedCallStatefulPartitionedCall tf.nn.relu_11/Relu:activations:0normalize_3_20518269normalize_3_20518271*
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
I__inference_normalize_3_layer_call_and_return_conditional_losses_205181452%
#normalize_3/StatefulPartitionedCall?
 dense_35/StatefulPartitionedCallStatefulPartitionedCall,normalize_3/StatefulPartitionedCall:output:0dense_35_20518274dense_35_20518276*
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
F__inference_dense_35_layer_call_and_return_conditional_losses_205181712"
 dense_35/StatefulPartitionedCall?
IdentityIdentity)dense_35/StatefulPartitionedCall:output:0!^dense_27/StatefulPartitionedCall!^dense_28/StatefulPartitionedCall!^dense_29/StatefulPartitionedCall!^dense_30/StatefulPartitionedCall!^dense_31/StatefulPartitionedCall!^dense_32/StatefulPartitionedCall!^dense_33/StatefulPartitionedCall!^dense_34/StatefulPartitionedCall!^dense_35/StatefulPartitionedCall ^model_6/StatefulPartitionedCall ^model_7/StatefulPartitionedCall$^normalize_3/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????:?????????:?????????
: : :::: :::: : ::::::::::::::::::::2D
 dense_27/StatefulPartitionedCall dense_27/StatefulPartitionedCall2D
 dense_28/StatefulPartitionedCall dense_28/StatefulPartitionedCall2D
 dense_29/StatefulPartitionedCall dense_29/StatefulPartitionedCall2D
 dense_30/StatefulPartitionedCall dense_30/StatefulPartitionedCall2D
 dense_31/StatefulPartitionedCall dense_31/StatefulPartitionedCall2D
 dense_32/StatefulPartitionedCall dense_32/StatefulPartitionedCall2D
 dense_33/StatefulPartitionedCall dense_33/StatefulPartitionedCall2D
 dense_34/StatefulPartitionedCall dense_34/StatefulPartitionedCall2D
 dense_35/StatefulPartitionedCall dense_35/StatefulPartitionedCall2B
model_6/StatefulPartitionedCallmodel_6/StatefulPartitionedCall2B
model_7/StatefulPartitionedCallmodel_7/StatefulPartitionedCall2J
#normalize_3/StatefulPartitionedCall#normalize_3/StatefulPartitionedCall:O K
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
?
+__inference_dense_27_layer_call_fn_20519400

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
F__inference_dense_27_layer_call_and_return_conditional_losses_205179212
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
?
u
/__inference_embedding_21_layer_call_fn_20519687

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
J__inference_embedding_21_layer_call_and_return_conditional_losses_205176572
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
?	
?
F__inference_dense_28_layer_call_and_return_conditional_losses_20519411

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
?-
?
E__inference_model_6_layer_call_and_return_conditional_losses_20517474
input_7*
&tf_math_greater_equal_9_greaterequal_y
embedding_20_20517418
embedding_18_20517440
embedding_19_20517464
identity??$embedding_18/StatefulPartitionedCall?$embedding_19/StatefulPartitionedCall?$embedding_20/StatefulPartitionedCall?
flatten_6/PartitionedCallPartitionedCallinput_7*
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
GPU2 *0J 8? *P
fKRI
G__inference_flatten_6_layer_call_and_return_conditional_losses_205173812
flatten_6/PartitionedCall?
*tf.clip_by_value_9/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 * $tI2,
*tf.clip_by_value_9/clip_by_value/Minimum/y?
(tf.clip_by_value_9/clip_by_value/MinimumMinimum"flatten_6/PartitionedCall:output:03tf.clip_by_value_9/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2*
(tf.clip_by_value_9/clip_by_value/Minimum?
"tf.clip_by_value_9/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"tf.clip_by_value_9/clip_by_value/y?
 tf.clip_by_value_9/clip_by_valueMaximum,tf.clip_by_value_9/clip_by_value/Minimum:z:0+tf.clip_by_value_9/clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????2"
 tf.clip_by_value_9/clip_by_value?
#tf.compat.v1.floor_div_6/FloorDiv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2%
#tf.compat.v1.floor_div_6/FloorDiv/y?
!tf.compat.v1.floor_div_6/FloorDivFloorDiv$tf.clip_by_value_9/clip_by_value:z:0,tf.compat.v1.floor_div_6/FloorDiv/y:output:0*
T0*'
_output_shapes
:?????????2#
!tf.compat.v1.floor_div_6/FloorDiv?
$tf.math.greater_equal_9/GreaterEqualGreaterEqual"flatten_6/PartitionedCall:output:0&tf_math_greater_equal_9_greaterequal_y*
T0*'
_output_shapes
:?????????2&
$tf.math.greater_equal_9/GreaterEqual?
tf.math.floormod_6/FloorMod/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2
tf.math.floormod_6/FloorMod/y?
tf.math.floormod_6/FloorModFloorMod$tf.clip_by_value_9/clip_by_value:z:0&tf.math.floormod_6/FloorMod/y:output:0*
T0*'
_output_shapes
:?????????2
tf.math.floormod_6/FloorMod?
$embedding_20/StatefulPartitionedCallStatefulPartitionedCall$tf.clip_by_value_9/clip_by_value:z:0embedding_20_20517418*
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
J__inference_embedding_20_layer_call_and_return_conditional_losses_205174092&
$embedding_20/StatefulPartitionedCall?
$embedding_18/StatefulPartitionedCallStatefulPartitionedCall%tf.compat.v1.floor_div_6/FloorDiv:z:0embedding_18_20517440*
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
J__inference_embedding_18_layer_call_and_return_conditional_losses_205174312&
$embedding_18/StatefulPartitionedCall?
tf.cast_9/CastCast(tf.math.greater_equal_9/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2
tf.cast_9/Cast?
tf.__operators__.add_18/AddV2AddV2-embedding_20/StatefulPartitionedCall:output:0-embedding_18/StatefulPartitionedCall:output:0*
T0*,
_output_shapes
:??????????2
tf.__operators__.add_18/AddV2?
$embedding_19/StatefulPartitionedCallStatefulPartitionedCalltf.math.floormod_6/FloorMod:z:0embedding_19_20517464*
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
J__inference_embedding_19_layer_call_and_return_conditional_losses_205174552&
$embedding_19/StatefulPartitionedCall?
tf.__operators__.add_19/AddV2AddV2!tf.__operators__.add_18/AddV2:z:0-embedding_19/StatefulPartitionedCall:output:0*
T0*,
_output_shapes
:??????????2
tf.__operators__.add_19/AddV2?
tf.expand_dims_6/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2!
tf.expand_dims_6/ExpandDims/dim?
tf.expand_dims_6/ExpandDims
ExpandDimstf.cast_9/Cast:y:0(tf.expand_dims_6/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????2
tf.expand_dims_6/ExpandDims?
tf.math.multiply_6/MulMul!tf.__operators__.add_19/AddV2:z:0$tf.expand_dims_6/ExpandDims:output:0*
T0*,
_output_shapes
:??????????2
tf.math.multiply_6/Mul?
*tf.math.reduce_sum_6/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2,
*tf.math.reduce_sum_6/Sum/reduction_indices?
tf.math.reduce_sum_6/SumSumtf.math.multiply_6/Mul:z:03tf.math.reduce_sum_6/Sum/reduction_indices:output:0*
T0*(
_output_shapes
:??????????2
tf.math.reduce_sum_6/Sum?
IdentityIdentity!tf.math.reduce_sum_6/Sum:output:0%^embedding_18/StatefulPartitionedCall%^embedding_19/StatefulPartitionedCall%^embedding_20/StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:?????????: :::2L
$embedding_18/StatefulPartitionedCall$embedding_18/StatefulPartitionedCall2L
$embedding_19/StatefulPartitionedCall$embedding_19/StatefulPartitionedCall2L
$embedding_20/StatefulPartitionedCall$embedding_20/StatefulPartitionedCall:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_7:

_output_shapes
: 
?
?
+__inference_dense_30_layer_call_fn_20519439

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
F__inference_dense_30_layer_call_and_return_conditional_losses_205179472
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
??
?
L__inference_custom_model_3_layer_call_and_return_conditional_losses_20519022

inputs_0_0

inputs_0_1
inputs_1+
'tf_math_greater_equal_11_greaterequal_y2
.model_6_tf_math_greater_equal_9_greaterequal_y2
.model_6_embedding_20_embedding_lookup_205188722
.model_6_embedding_18_embedding_lookup_205188782
.model_6_embedding_19_embedding_lookup_205188863
/model_7_tf_math_greater_equal_10_greaterequal_y2
.model_7_embedding_23_embedding_lookup_205189102
.model_7_embedding_21_embedding_lookup_205189162
.model_7_embedding_22_embedding_lookup_20518924/
+tf_clip_by_value_11_clip_by_value_minimum_y'
#tf_clip_by_value_11_clip_by_value_y+
'dense_27_matmul_readvariableop_resource,
(dense_27_biasadd_readvariableop_resource+
'dense_30_matmul_readvariableop_resource,
(dense_30_biasadd_readvariableop_resource+
'dense_28_matmul_readvariableop_resource,
(dense_28_biasadd_readvariableop_resource+
'dense_29_matmul_readvariableop_resource,
(dense_29_biasadd_readvariableop_resource+
'dense_31_matmul_readvariableop_resource,
(dense_31_biasadd_readvariableop_resource+
'dense_32_matmul_readvariableop_resource,
(dense_32_biasadd_readvariableop_resource+
'dense_33_matmul_readvariableop_resource,
(dense_33_biasadd_readvariableop_resource+
'dense_34_matmul_readvariableop_resource,
(dense_34_biasadd_readvariableop_resource?
;normalize_3_normalization_3_reshape_readvariableop_resourceA
=normalize_3_normalization_3_reshape_1_readvariableop_resource+
'dense_35_matmul_readvariableop_resource,
(dense_35_biasadd_readvariableop_resource
identity??dense_27/BiasAdd/ReadVariableOp?dense_27/MatMul/ReadVariableOp?dense_28/BiasAdd/ReadVariableOp?dense_28/MatMul/ReadVariableOp?dense_29/BiasAdd/ReadVariableOp?dense_29/MatMul/ReadVariableOp?dense_30/BiasAdd/ReadVariableOp?dense_30/MatMul/ReadVariableOp?dense_31/BiasAdd/ReadVariableOp?dense_31/MatMul/ReadVariableOp?dense_32/BiasAdd/ReadVariableOp?dense_32/MatMul/ReadVariableOp?dense_33/BiasAdd/ReadVariableOp?dense_33/MatMul/ReadVariableOp?dense_34/BiasAdd/ReadVariableOp?dense_34/MatMul/ReadVariableOp?dense_35/BiasAdd/ReadVariableOp?dense_35/MatMul/ReadVariableOp?%model_6/embedding_18/embedding_lookup?%model_6/embedding_19/embedding_lookup?%model_6/embedding_20/embedding_lookup?%model_7/embedding_21/embedding_lookup?%model_7/embedding_22/embedding_lookup?%model_7/embedding_23/embedding_lookup?2normalize_3/normalization_3/Reshape/ReadVariableOp?4normalize_3/normalization_3/Reshape_1/ReadVariableOp?
%tf.math.greater_equal_11/GreaterEqualGreaterEqualinputs_1'tf_math_greater_equal_11_greaterequal_y*
T0*'
_output_shapes
:?????????
2'
%tf.math.greater_equal_11/GreaterEqual?
model_6/flatten_6/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
model_6/flatten_6/Const?
model_6/flatten_6/ReshapeReshape
inputs_0_0 model_6/flatten_6/Const:output:0*
T0*'
_output_shapes
:?????????2
model_6/flatten_6/Reshape?
2model_6/tf.clip_by_value_9/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 * $tI24
2model_6/tf.clip_by_value_9/clip_by_value/Minimum/y?
0model_6/tf.clip_by_value_9/clip_by_value/MinimumMinimum"model_6/flatten_6/Reshape:output:0;model_6/tf.clip_by_value_9/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????22
0model_6/tf.clip_by_value_9/clip_by_value/Minimum?
*model_6/tf.clip_by_value_9/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2,
*model_6/tf.clip_by_value_9/clip_by_value/y?
(model_6/tf.clip_by_value_9/clip_by_valueMaximum4model_6/tf.clip_by_value_9/clip_by_value/Minimum:z:03model_6/tf.clip_by_value_9/clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????2*
(model_6/tf.clip_by_value_9/clip_by_value?
+model_6/tf.compat.v1.floor_div_6/FloorDiv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2-
+model_6/tf.compat.v1.floor_div_6/FloorDiv/y?
)model_6/tf.compat.v1.floor_div_6/FloorDivFloorDiv,model_6/tf.clip_by_value_9/clip_by_value:z:04model_6/tf.compat.v1.floor_div_6/FloorDiv/y:output:0*
T0*'
_output_shapes
:?????????2+
)model_6/tf.compat.v1.floor_div_6/FloorDiv?
,model_6/tf.math.greater_equal_9/GreaterEqualGreaterEqual"model_6/flatten_6/Reshape:output:0.model_6_tf_math_greater_equal_9_greaterequal_y*
T0*'
_output_shapes
:?????????2.
,model_6/tf.math.greater_equal_9/GreaterEqual?
%model_6/tf.math.floormod_6/FloorMod/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2'
%model_6/tf.math.floormod_6/FloorMod/y?
#model_6/tf.math.floormod_6/FloorModFloorMod,model_6/tf.clip_by_value_9/clip_by_value:z:0.model_6/tf.math.floormod_6/FloorMod/y:output:0*
T0*'
_output_shapes
:?????????2%
#model_6/tf.math.floormod_6/FloorMod?
model_6/embedding_20/CastCast,model_6/tf.clip_by_value_9/clip_by_value:z:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2
model_6/embedding_20/Cast?
%model_6/embedding_20/embedding_lookupResourceGather.model_6_embedding_20_embedding_lookup_20518872model_6/embedding_20/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*A
_class7
53loc:@model_6/embedding_20/embedding_lookup/20518872*,
_output_shapes
:??????????*
dtype02'
%model_6/embedding_20/embedding_lookup?
.model_6/embedding_20/embedding_lookup/IdentityIdentity.model_6/embedding_20/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*A
_class7
53loc:@model_6/embedding_20/embedding_lookup/20518872*,
_output_shapes
:??????????20
.model_6/embedding_20/embedding_lookup/Identity?
0model_6/embedding_20/embedding_lookup/Identity_1Identity7model_6/embedding_20/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????22
0model_6/embedding_20/embedding_lookup/Identity_1?
model_6/embedding_18/CastCast-model_6/tf.compat.v1.floor_div_6/FloorDiv:z:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2
model_6/embedding_18/Cast?
%model_6/embedding_18/embedding_lookupResourceGather.model_6_embedding_18_embedding_lookup_20518878model_6/embedding_18/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*A
_class7
53loc:@model_6/embedding_18/embedding_lookup/20518878*,
_output_shapes
:??????????*
dtype02'
%model_6/embedding_18/embedding_lookup?
.model_6/embedding_18/embedding_lookup/IdentityIdentity.model_6/embedding_18/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*A
_class7
53loc:@model_6/embedding_18/embedding_lookup/20518878*,
_output_shapes
:??????????20
.model_6/embedding_18/embedding_lookup/Identity?
0model_6/embedding_18/embedding_lookup/Identity_1Identity7model_6/embedding_18/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????22
0model_6/embedding_18/embedding_lookup/Identity_1?
model_6/tf.cast_9/CastCast0model_6/tf.math.greater_equal_9/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2
model_6/tf.cast_9/Cast?
%model_6/tf.__operators__.add_18/AddV2AddV29model_6/embedding_20/embedding_lookup/Identity_1:output:09model_6/embedding_18/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:??????????2'
%model_6/tf.__operators__.add_18/AddV2?
model_6/embedding_19/CastCast'model_6/tf.math.floormod_6/FloorMod:z:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2
model_6/embedding_19/Cast?
%model_6/embedding_19/embedding_lookupResourceGather.model_6_embedding_19_embedding_lookup_20518886model_6/embedding_19/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*A
_class7
53loc:@model_6/embedding_19/embedding_lookup/20518886*,
_output_shapes
:??????????*
dtype02'
%model_6/embedding_19/embedding_lookup?
.model_6/embedding_19/embedding_lookup/IdentityIdentity.model_6/embedding_19/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*A
_class7
53loc:@model_6/embedding_19/embedding_lookup/20518886*,
_output_shapes
:??????????20
.model_6/embedding_19/embedding_lookup/Identity?
0model_6/embedding_19/embedding_lookup/Identity_1Identity7model_6/embedding_19/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????22
0model_6/embedding_19/embedding_lookup/Identity_1?
%model_6/tf.__operators__.add_19/AddV2AddV2)model_6/tf.__operators__.add_18/AddV2:z:09model_6/embedding_19/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:??????????2'
%model_6/tf.__operators__.add_19/AddV2?
'model_6/tf.expand_dims_6/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2)
'model_6/tf.expand_dims_6/ExpandDims/dim?
#model_6/tf.expand_dims_6/ExpandDims
ExpandDimsmodel_6/tf.cast_9/Cast:y:00model_6/tf.expand_dims_6/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????2%
#model_6/tf.expand_dims_6/ExpandDims?
model_6/tf.math.multiply_6/MulMul)model_6/tf.__operators__.add_19/AddV2:z:0,model_6/tf.expand_dims_6/ExpandDims:output:0*
T0*,
_output_shapes
:??????????2 
model_6/tf.math.multiply_6/Mul?
2model_6/tf.math.reduce_sum_6/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :24
2model_6/tf.math.reduce_sum_6/Sum/reduction_indices?
 model_6/tf.math.reduce_sum_6/SumSum"model_6/tf.math.multiply_6/Mul:z:0;model_6/tf.math.reduce_sum_6/Sum/reduction_indices:output:0*
T0*(
_output_shapes
:??????????2"
 model_6/tf.math.reduce_sum_6/Sum?
model_7/flatten_7/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
model_7/flatten_7/Const?
model_7/flatten_7/ReshapeReshape
inputs_0_1 model_7/flatten_7/Const:output:0*
T0*'
_output_shapes
:?????????2
model_7/flatten_7/Reshape?
3model_7/tf.clip_by_value_10/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 * $tI25
3model_7/tf.clip_by_value_10/clip_by_value/Minimum/y?
1model_7/tf.clip_by_value_10/clip_by_value/MinimumMinimum"model_7/flatten_7/Reshape:output:0<model_7/tf.clip_by_value_10/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????23
1model_7/tf.clip_by_value_10/clip_by_value/Minimum?
+model_7/tf.clip_by_value_10/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2-
+model_7/tf.clip_by_value_10/clip_by_value/y?
)model_7/tf.clip_by_value_10/clip_by_valueMaximum5model_7/tf.clip_by_value_10/clip_by_value/Minimum:z:04model_7/tf.clip_by_value_10/clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????2+
)model_7/tf.clip_by_value_10/clip_by_value?
+model_7/tf.compat.v1.floor_div_7/FloorDiv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2-
+model_7/tf.compat.v1.floor_div_7/FloorDiv/y?
)model_7/tf.compat.v1.floor_div_7/FloorDivFloorDiv-model_7/tf.clip_by_value_10/clip_by_value:z:04model_7/tf.compat.v1.floor_div_7/FloorDiv/y:output:0*
T0*'
_output_shapes
:?????????2+
)model_7/tf.compat.v1.floor_div_7/FloorDiv?
-model_7/tf.math.greater_equal_10/GreaterEqualGreaterEqual"model_7/flatten_7/Reshape:output:0/model_7_tf_math_greater_equal_10_greaterequal_y*
T0*'
_output_shapes
:?????????2/
-model_7/tf.math.greater_equal_10/GreaterEqual?
%model_7/tf.math.floormod_7/FloorMod/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2'
%model_7/tf.math.floormod_7/FloorMod/y?
#model_7/tf.math.floormod_7/FloorModFloorMod-model_7/tf.clip_by_value_10/clip_by_value:z:0.model_7/tf.math.floormod_7/FloorMod/y:output:0*
T0*'
_output_shapes
:?????????2%
#model_7/tf.math.floormod_7/FloorMod?
model_7/embedding_23/CastCast-model_7/tf.clip_by_value_10/clip_by_value:z:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2
model_7/embedding_23/Cast?
%model_7/embedding_23/embedding_lookupResourceGather.model_7_embedding_23_embedding_lookup_20518910model_7/embedding_23/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*A
_class7
53loc:@model_7/embedding_23/embedding_lookup/20518910*,
_output_shapes
:??????????*
dtype02'
%model_7/embedding_23/embedding_lookup?
.model_7/embedding_23/embedding_lookup/IdentityIdentity.model_7/embedding_23/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*A
_class7
53loc:@model_7/embedding_23/embedding_lookup/20518910*,
_output_shapes
:??????????20
.model_7/embedding_23/embedding_lookup/Identity?
0model_7/embedding_23/embedding_lookup/Identity_1Identity7model_7/embedding_23/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????22
0model_7/embedding_23/embedding_lookup/Identity_1?
model_7/embedding_21/CastCast-model_7/tf.compat.v1.floor_div_7/FloorDiv:z:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2
model_7/embedding_21/Cast?
%model_7/embedding_21/embedding_lookupResourceGather.model_7_embedding_21_embedding_lookup_20518916model_7/embedding_21/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*A
_class7
53loc:@model_7/embedding_21/embedding_lookup/20518916*,
_output_shapes
:??????????*
dtype02'
%model_7/embedding_21/embedding_lookup?
.model_7/embedding_21/embedding_lookup/IdentityIdentity.model_7/embedding_21/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*A
_class7
53loc:@model_7/embedding_21/embedding_lookup/20518916*,
_output_shapes
:??????????20
.model_7/embedding_21/embedding_lookup/Identity?
0model_7/embedding_21/embedding_lookup/Identity_1Identity7model_7/embedding_21/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????22
0model_7/embedding_21/embedding_lookup/Identity_1?
model_7/tf.cast_10/CastCast1model_7/tf.math.greater_equal_10/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2
model_7/tf.cast_10/Cast?
%model_7/tf.__operators__.add_20/AddV2AddV29model_7/embedding_23/embedding_lookup/Identity_1:output:09model_7/embedding_21/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:??????????2'
%model_7/tf.__operators__.add_20/AddV2?
model_7/embedding_22/CastCast'model_7/tf.math.floormod_7/FloorMod:z:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2
model_7/embedding_22/Cast?
%model_7/embedding_22/embedding_lookupResourceGather.model_7_embedding_22_embedding_lookup_20518924model_7/embedding_22/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*A
_class7
53loc:@model_7/embedding_22/embedding_lookup/20518924*,
_output_shapes
:??????????*
dtype02'
%model_7/embedding_22/embedding_lookup?
.model_7/embedding_22/embedding_lookup/IdentityIdentity.model_7/embedding_22/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*A
_class7
53loc:@model_7/embedding_22/embedding_lookup/20518924*,
_output_shapes
:??????????20
.model_7/embedding_22/embedding_lookup/Identity?
0model_7/embedding_22/embedding_lookup/Identity_1Identity7model_7/embedding_22/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????22
0model_7/embedding_22/embedding_lookup/Identity_1?
%model_7/tf.__operators__.add_21/AddV2AddV2)model_7/tf.__operators__.add_20/AddV2:z:09model_7/embedding_22/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:??????????2'
%model_7/tf.__operators__.add_21/AddV2?
'model_7/tf.expand_dims_7/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2)
'model_7/tf.expand_dims_7/ExpandDims/dim?
#model_7/tf.expand_dims_7/ExpandDims
ExpandDimsmodel_7/tf.cast_10/Cast:y:00model_7/tf.expand_dims_7/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????2%
#model_7/tf.expand_dims_7/ExpandDims?
model_7/tf.math.multiply_7/MulMul)model_7/tf.__operators__.add_21/AddV2:z:0,model_7/tf.expand_dims_7/ExpandDims:output:0*
T0*,
_output_shapes
:??????????2 
model_7/tf.math.multiply_7/Mul?
2model_7/tf.math.reduce_sum_7/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :24
2model_7/tf.math.reduce_sum_7/Sum/reduction_indices?
 model_7/tf.math.reduce_sum_7/SumSum"model_7/tf.math.multiply_7/Mul:z:0;model_7/tf.math.reduce_sum_7/Sum/reduction_indices:output:0*
T0*(
_output_shapes
:??????????2"
 model_7/tf.math.reduce_sum_7/Sum?
)tf.clip_by_value_11/clip_by_value/MinimumMinimuminputs_1+tf_clip_by_value_11_clip_by_value_minimum_y*
T0*'
_output_shapes
:?????????
2+
)tf.clip_by_value_11/clip_by_value/Minimum?
!tf.clip_by_value_11/clip_by_valueMaximum-tf.clip_by_value_11/clip_by_value/Minimum:z:0#tf_clip_by_value_11_clip_by_value_y*
T0*'
_output_shapes
:?????????
2#
!tf.clip_by_value_11/clip_by_value?
tf.cast_11/CastCast)tf.math.greater_equal_11/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????
2
tf.cast_11/Castt
tf.concat_9/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
tf.concat_9/concat/axis?
tf.concat_9/concatConcatV2)model_6/tf.math.reduce_sum_6/Sum:output:0)model_7/tf.math.reduce_sum_7/Sum:output:0 tf.concat_9/concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????2
tf.concat_9/concat
tf.concat_10/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
tf.concat_10/concat/axis?
tf.concat_10/concatConcatV2%tf.clip_by_value_11/clip_by_value:z:0tf.cast_11/Cast:y:0!tf.concat_10/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????2
tf.concat_10/concat?
dense_27/MatMul/ReadVariableOpReadVariableOp'dense_27_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02 
dense_27/MatMul/ReadVariableOp?
dense_27/MatMulMatMultf.concat_9/concat:output:0&dense_27/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_27/MatMul?
dense_27/BiasAdd/ReadVariableOpReadVariableOp(dense_27_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
dense_27/BiasAdd/ReadVariableOp?
dense_27/BiasAddBiasAdddense_27/MatMul:product:0'dense_27/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_27/BiasAddt
dense_27/ReluReludense_27/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_27/Relu?
dense_30/MatMul/ReadVariableOpReadVariableOp'dense_30_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02 
dense_30/MatMul/ReadVariableOp?
dense_30/MatMulMatMultf.concat_10/concat:output:0&dense_30/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_30/MatMul?
dense_30/BiasAdd/ReadVariableOpReadVariableOp(dense_30_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
dense_30/BiasAdd/ReadVariableOp?
dense_30/BiasAddBiasAdddense_30/MatMul:product:0'dense_30/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_30/BiasAdd?
dense_28/MatMul/ReadVariableOpReadVariableOp'dense_28_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02 
dense_28/MatMul/ReadVariableOp?
dense_28/MatMulMatMuldense_27/Relu:activations:0&dense_28/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_28/MatMul?
dense_28/BiasAdd/ReadVariableOpReadVariableOp(dense_28_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
dense_28/BiasAdd/ReadVariableOp?
dense_28/BiasAddBiasAdddense_28/MatMul:product:0'dense_28/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_28/BiasAddt
dense_28/ReluReludense_28/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_28/Relu?
dense_29/MatMul/ReadVariableOpReadVariableOp'dense_29_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02 
dense_29/MatMul/ReadVariableOp?
dense_29/MatMulMatMuldense_28/Relu:activations:0&dense_29/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_29/MatMul?
dense_29/BiasAdd/ReadVariableOpReadVariableOp(dense_29_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
dense_29/BiasAdd/ReadVariableOp?
dense_29/BiasAddBiasAdddense_29/MatMul:product:0'dense_29/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_29/BiasAddt
dense_29/ReluReludense_29/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_29/Relu?
dense_31/MatMul/ReadVariableOpReadVariableOp'dense_31_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02 
dense_31/MatMul/ReadVariableOp?
dense_31/MatMulMatMuldense_30/BiasAdd:output:0&dense_31/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_31/MatMul?
dense_31/BiasAdd/ReadVariableOpReadVariableOp(dense_31_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
dense_31/BiasAdd/ReadVariableOp?
dense_31/BiasAddBiasAdddense_31/MatMul:product:0'dense_31/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_31/BiasAdd
tf.concat_11/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
tf.concat_11/concat/axis?
tf.concat_11/concatConcatV2dense_29/Relu:activations:0dense_31/BiasAdd:output:0!tf.concat_11/concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????2
tf.concat_11/concat?
dense_32/MatMul/ReadVariableOpReadVariableOp'dense_32_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02 
dense_32/MatMul/ReadVariableOp?
dense_32/MatMulMatMultf.concat_11/concat:output:0&dense_32/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_32/MatMul?
dense_32/BiasAdd/ReadVariableOpReadVariableOp(dense_32_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
dense_32/BiasAdd/ReadVariableOp?
dense_32/BiasAddBiasAdddense_32/MatMul:product:0'dense_32/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_32/BiasAdd|
tf.nn.relu_9/ReluReludense_32/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
tf.nn.relu_9/Relu?
dense_33/MatMul/ReadVariableOpReadVariableOp'dense_33_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02 
dense_33/MatMul/ReadVariableOp?
dense_33/MatMulMatMultf.nn.relu_9/Relu:activations:0&dense_33/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_33/MatMul?
dense_33/BiasAdd/ReadVariableOpReadVariableOp(dense_33_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
dense_33/BiasAdd/ReadVariableOp?
dense_33/BiasAddBiasAdddense_33/MatMul:product:0'dense_33/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_33/BiasAdd?
tf.__operators__.add_22/AddV2AddV2dense_33/BiasAdd:output:0tf.nn.relu_9/Relu:activations:0*
T0*(
_output_shapes
:??????????2
tf.__operators__.add_22/AddV2?
tf.nn.relu_10/ReluRelu!tf.__operators__.add_22/AddV2:z:0*
T0*(
_output_shapes
:??????????2
tf.nn.relu_10/Relu?
dense_34/MatMul/ReadVariableOpReadVariableOp'dense_34_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02 
dense_34/MatMul/ReadVariableOp?
dense_34/MatMulMatMul tf.nn.relu_10/Relu:activations:0&dense_34/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_34/MatMul?
dense_34/BiasAdd/ReadVariableOpReadVariableOp(dense_34_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
dense_34/BiasAdd/ReadVariableOp?
dense_34/BiasAddBiasAdddense_34/MatMul:product:0'dense_34/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_34/BiasAdd?
tf.__operators__.add_23/AddV2AddV2dense_34/BiasAdd:output:0 tf.nn.relu_10/Relu:activations:0*
T0*(
_output_shapes
:??????????2
tf.__operators__.add_23/AddV2?
tf.nn.relu_11/ReluRelu!tf.__operators__.add_23/AddV2:z:0*
T0*(
_output_shapes
:??????????2
tf.nn.relu_11/Relu?
2normalize_3/normalization_3/Reshape/ReadVariableOpReadVariableOp;normalize_3_normalization_3_reshape_readvariableop_resource*
_output_shapes	
:?*
dtype024
2normalize_3/normalization_3/Reshape/ReadVariableOp?
)normalize_3/normalization_3/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2+
)normalize_3/normalization_3/Reshape/shape?
#normalize_3/normalization_3/ReshapeReshape:normalize_3/normalization_3/Reshape/ReadVariableOp:value:02normalize_3/normalization_3/Reshape/shape:output:0*
T0*
_output_shapes
:	?2%
#normalize_3/normalization_3/Reshape?
4normalize_3/normalization_3/Reshape_1/ReadVariableOpReadVariableOp=normalize_3_normalization_3_reshape_1_readvariableop_resource*
_output_shapes	
:?*
dtype026
4normalize_3/normalization_3/Reshape_1/ReadVariableOp?
+normalize_3/normalization_3/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2-
+normalize_3/normalization_3/Reshape_1/shape?
%normalize_3/normalization_3/Reshape_1Reshape<normalize_3/normalization_3/Reshape_1/ReadVariableOp:value:04normalize_3/normalization_3/Reshape_1/shape:output:0*
T0*
_output_shapes
:	?2'
%normalize_3/normalization_3/Reshape_1?
normalize_3/normalization_3/subSub tf.nn.relu_11/Relu:activations:0,normalize_3/normalization_3/Reshape:output:0*
T0*(
_output_shapes
:??????????2!
normalize_3/normalization_3/sub?
 normalize_3/normalization_3/SqrtSqrt.normalize_3/normalization_3/Reshape_1:output:0*
T0*
_output_shapes
:	?2"
 normalize_3/normalization_3/Sqrt?
%normalize_3/normalization_3/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32'
%normalize_3/normalization_3/Maximum/y?
#normalize_3/normalization_3/MaximumMaximum$normalize_3/normalization_3/Sqrt:y:0.normalize_3/normalization_3/Maximum/y:output:0*
T0*
_output_shapes
:	?2%
#normalize_3/normalization_3/Maximum?
#normalize_3/normalization_3/truedivRealDiv#normalize_3/normalization_3/sub:z:0'normalize_3/normalization_3/Maximum:z:0*
T0*(
_output_shapes
:??????????2%
#normalize_3/normalization_3/truediv?
dense_35/MatMul/ReadVariableOpReadVariableOp'dense_35_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02 
dense_35/MatMul/ReadVariableOp?
dense_35/MatMulMatMul'normalize_3/normalization_3/truediv:z:0&dense_35/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_35/MatMul?
dense_35/BiasAdd/ReadVariableOpReadVariableOp(dense_35_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_35/BiasAdd/ReadVariableOp?
dense_35/BiasAddBiasAdddense_35/MatMul:product:0'dense_35/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_35/BiasAdd?
IdentityIdentitydense_35/BiasAdd:output:0 ^dense_27/BiasAdd/ReadVariableOp^dense_27/MatMul/ReadVariableOp ^dense_28/BiasAdd/ReadVariableOp^dense_28/MatMul/ReadVariableOp ^dense_29/BiasAdd/ReadVariableOp^dense_29/MatMul/ReadVariableOp ^dense_30/BiasAdd/ReadVariableOp^dense_30/MatMul/ReadVariableOp ^dense_31/BiasAdd/ReadVariableOp^dense_31/MatMul/ReadVariableOp ^dense_32/BiasAdd/ReadVariableOp^dense_32/MatMul/ReadVariableOp ^dense_33/BiasAdd/ReadVariableOp^dense_33/MatMul/ReadVariableOp ^dense_34/BiasAdd/ReadVariableOp^dense_34/MatMul/ReadVariableOp ^dense_35/BiasAdd/ReadVariableOp^dense_35/MatMul/ReadVariableOp&^model_6/embedding_18/embedding_lookup&^model_6/embedding_19/embedding_lookup&^model_6/embedding_20/embedding_lookup&^model_7/embedding_21/embedding_lookup&^model_7/embedding_22/embedding_lookup&^model_7/embedding_23/embedding_lookup3^normalize_3/normalization_3/Reshape/ReadVariableOp5^normalize_3/normalization_3/Reshape_1/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????:?????????:?????????
: : :::: :::: : ::::::::::::::::::::2B
dense_27/BiasAdd/ReadVariableOpdense_27/BiasAdd/ReadVariableOp2@
dense_27/MatMul/ReadVariableOpdense_27/MatMul/ReadVariableOp2B
dense_28/BiasAdd/ReadVariableOpdense_28/BiasAdd/ReadVariableOp2@
dense_28/MatMul/ReadVariableOpdense_28/MatMul/ReadVariableOp2B
dense_29/BiasAdd/ReadVariableOpdense_29/BiasAdd/ReadVariableOp2@
dense_29/MatMul/ReadVariableOpdense_29/MatMul/ReadVariableOp2B
dense_30/BiasAdd/ReadVariableOpdense_30/BiasAdd/ReadVariableOp2@
dense_30/MatMul/ReadVariableOpdense_30/MatMul/ReadVariableOp2B
dense_31/BiasAdd/ReadVariableOpdense_31/BiasAdd/ReadVariableOp2@
dense_31/MatMul/ReadVariableOpdense_31/MatMul/ReadVariableOp2B
dense_32/BiasAdd/ReadVariableOpdense_32/BiasAdd/ReadVariableOp2@
dense_32/MatMul/ReadVariableOpdense_32/MatMul/ReadVariableOp2B
dense_33/BiasAdd/ReadVariableOpdense_33/BiasAdd/ReadVariableOp2@
dense_33/MatMul/ReadVariableOpdense_33/MatMul/ReadVariableOp2B
dense_34/BiasAdd/ReadVariableOpdense_34/BiasAdd/ReadVariableOp2@
dense_34/MatMul/ReadVariableOpdense_34/MatMul/ReadVariableOp2B
dense_35/BiasAdd/ReadVariableOpdense_35/BiasAdd/ReadVariableOp2@
dense_35/MatMul/ReadVariableOpdense_35/MatMul/ReadVariableOp2N
%model_6/embedding_18/embedding_lookup%model_6/embedding_18/embedding_lookup2N
%model_6/embedding_19/embedding_lookup%model_6/embedding_19/embedding_lookup2N
%model_6/embedding_20/embedding_lookup%model_6/embedding_20/embedding_lookup2N
%model_7/embedding_21/embedding_lookup%model_7/embedding_21/embedding_lookup2N
%model_7/embedding_22/embedding_lookup%model_7/embedding_22/embedding_lookup2N
%model_7/embedding_23/embedding_lookup%model_7/embedding_23/embedding_lookup2h
2normalize_3/normalization_3/Reshape/ReadVariableOp2normalize_3/normalization_3/Reshape/ReadVariableOp2l
4normalize_3/normalization_3/Reshape_1/ReadVariableOp4normalize_3/normalization_3/Reshape_1/ReadVariableOp:S O
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
J__inference_embedding_20_layer_call_and_return_conditional_losses_20519601

inputs
embedding_lookup_20519595
identity??embedding_lookup]
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:?????????2
Cast?
embedding_lookupResourceGatherembedding_lookup_20519595Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*,
_class"
 loc:@embedding_lookup/20519595*,
_output_shapes
:??????????*
dtype02
embedding_lookup?
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*,
_class"
 loc:@embedding_lookup/20519595*,
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
?8
?
E__inference_model_6_layer_call_and_return_conditional_losses_20519244

inputs*
&tf_math_greater_equal_9_greaterequal_y*
&embedding_20_embedding_lookup_20519218*
&embedding_18_embedding_lookup_20519224*
&embedding_19_embedding_lookup_20519232
identity??embedding_18/embedding_lookup?embedding_19/embedding_lookup?embedding_20/embedding_lookups
flatten_6/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
flatten_6/Const?
flatten_6/ReshapeReshapeinputsflatten_6/Const:output:0*
T0*'
_output_shapes
:?????????2
flatten_6/Reshape?
*tf.clip_by_value_9/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 * $tI2,
*tf.clip_by_value_9/clip_by_value/Minimum/y?
(tf.clip_by_value_9/clip_by_value/MinimumMinimumflatten_6/Reshape:output:03tf.clip_by_value_9/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2*
(tf.clip_by_value_9/clip_by_value/Minimum?
"tf.clip_by_value_9/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"tf.clip_by_value_9/clip_by_value/y?
 tf.clip_by_value_9/clip_by_valueMaximum,tf.clip_by_value_9/clip_by_value/Minimum:z:0+tf.clip_by_value_9/clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????2"
 tf.clip_by_value_9/clip_by_value?
#tf.compat.v1.floor_div_6/FloorDiv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2%
#tf.compat.v1.floor_div_6/FloorDiv/y?
!tf.compat.v1.floor_div_6/FloorDivFloorDiv$tf.clip_by_value_9/clip_by_value:z:0,tf.compat.v1.floor_div_6/FloorDiv/y:output:0*
T0*'
_output_shapes
:?????????2#
!tf.compat.v1.floor_div_6/FloorDiv?
$tf.math.greater_equal_9/GreaterEqualGreaterEqualflatten_6/Reshape:output:0&tf_math_greater_equal_9_greaterequal_y*
T0*'
_output_shapes
:?????????2&
$tf.math.greater_equal_9/GreaterEqual?
tf.math.floormod_6/FloorMod/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2
tf.math.floormod_6/FloorMod/y?
tf.math.floormod_6/FloorModFloorMod$tf.clip_by_value_9/clip_by_value:z:0&tf.math.floormod_6/FloorMod/y:output:0*
T0*'
_output_shapes
:?????????2
tf.math.floormod_6/FloorMod?
embedding_20/CastCast$tf.clip_by_value_9/clip_by_value:z:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2
embedding_20/Cast?
embedding_20/embedding_lookupResourceGather&embedding_20_embedding_lookup_20519218embedding_20/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*9
_class/
-+loc:@embedding_20/embedding_lookup/20519218*,
_output_shapes
:??????????*
dtype02
embedding_20/embedding_lookup?
&embedding_20/embedding_lookup/IdentityIdentity&embedding_20/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*9
_class/
-+loc:@embedding_20/embedding_lookup/20519218*,
_output_shapes
:??????????2(
&embedding_20/embedding_lookup/Identity?
(embedding_20/embedding_lookup/Identity_1Identity/embedding_20/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????2*
(embedding_20/embedding_lookup/Identity_1?
embedding_18/CastCast%tf.compat.v1.floor_div_6/FloorDiv:z:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2
embedding_18/Cast?
embedding_18/embedding_lookupResourceGather&embedding_18_embedding_lookup_20519224embedding_18/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*9
_class/
-+loc:@embedding_18/embedding_lookup/20519224*,
_output_shapes
:??????????*
dtype02
embedding_18/embedding_lookup?
&embedding_18/embedding_lookup/IdentityIdentity&embedding_18/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*9
_class/
-+loc:@embedding_18/embedding_lookup/20519224*,
_output_shapes
:??????????2(
&embedding_18/embedding_lookup/Identity?
(embedding_18/embedding_lookup/Identity_1Identity/embedding_18/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????2*
(embedding_18/embedding_lookup/Identity_1?
tf.cast_9/CastCast(tf.math.greater_equal_9/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2
tf.cast_9/Cast?
tf.__operators__.add_18/AddV2AddV21embedding_20/embedding_lookup/Identity_1:output:01embedding_18/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:??????????2
tf.__operators__.add_18/AddV2?
embedding_19/CastCasttf.math.floormod_6/FloorMod:z:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2
embedding_19/Cast?
embedding_19/embedding_lookupResourceGather&embedding_19_embedding_lookup_20519232embedding_19/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*9
_class/
-+loc:@embedding_19/embedding_lookup/20519232*,
_output_shapes
:??????????*
dtype02
embedding_19/embedding_lookup?
&embedding_19/embedding_lookup/IdentityIdentity&embedding_19/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*9
_class/
-+loc:@embedding_19/embedding_lookup/20519232*,
_output_shapes
:??????????2(
&embedding_19/embedding_lookup/Identity?
(embedding_19/embedding_lookup/Identity_1Identity/embedding_19/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????2*
(embedding_19/embedding_lookup/Identity_1?
tf.__operators__.add_19/AddV2AddV2!tf.__operators__.add_18/AddV2:z:01embedding_19/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:??????????2
tf.__operators__.add_19/AddV2?
tf.expand_dims_6/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2!
tf.expand_dims_6/ExpandDims/dim?
tf.expand_dims_6/ExpandDims
ExpandDimstf.cast_9/Cast:y:0(tf.expand_dims_6/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????2
tf.expand_dims_6/ExpandDims?
tf.math.multiply_6/MulMul!tf.__operators__.add_19/AddV2:z:0$tf.expand_dims_6/ExpandDims:output:0*
T0*,
_output_shapes
:??????????2
tf.math.multiply_6/Mul?
*tf.math.reduce_sum_6/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2,
*tf.math.reduce_sum_6/Sum/reduction_indices?
tf.math.reduce_sum_6/SumSumtf.math.multiply_6/Mul:z:03tf.math.reduce_sum_6/Sum/reduction_indices:output:0*
T0*(
_output_shapes
:??????????2
tf.math.reduce_sum_6/Sum?
IdentityIdentity!tf.math.reduce_sum_6/Sum:output:0^embedding_18/embedding_lookup^embedding_19/embedding_lookup^embedding_20/embedding_lookup*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:?????????: :::2>
embedding_18/embedding_lookupembedding_18/embedding_lookup2>
embedding_19/embedding_lookupembedding_19/embedding_lookup2>
embedding_20/embedding_lookupembedding_20/embedding_lookup:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: 
?	
?
F__inference_dense_31_layer_call_and_return_conditional_losses_20518027

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
F__inference_dense_33_layer_call_and_return_conditional_losses_20518082

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
?-
?
E__inference_model_6_layer_call_and_return_conditional_losses_20517506
input_7*
&tf_math_greater_equal_9_greaterequal_y
embedding_20_20517488
embedding_18_20517491
embedding_19_20517496
identity??$embedding_18/StatefulPartitionedCall?$embedding_19/StatefulPartitionedCall?$embedding_20/StatefulPartitionedCall?
flatten_6/PartitionedCallPartitionedCallinput_7*
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
GPU2 *0J 8? *P
fKRI
G__inference_flatten_6_layer_call_and_return_conditional_losses_205173812
flatten_6/PartitionedCall?
*tf.clip_by_value_9/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 * $tI2,
*tf.clip_by_value_9/clip_by_value/Minimum/y?
(tf.clip_by_value_9/clip_by_value/MinimumMinimum"flatten_6/PartitionedCall:output:03tf.clip_by_value_9/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2*
(tf.clip_by_value_9/clip_by_value/Minimum?
"tf.clip_by_value_9/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"tf.clip_by_value_9/clip_by_value/y?
 tf.clip_by_value_9/clip_by_valueMaximum,tf.clip_by_value_9/clip_by_value/Minimum:z:0+tf.clip_by_value_9/clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????2"
 tf.clip_by_value_9/clip_by_value?
#tf.compat.v1.floor_div_6/FloorDiv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2%
#tf.compat.v1.floor_div_6/FloorDiv/y?
!tf.compat.v1.floor_div_6/FloorDivFloorDiv$tf.clip_by_value_9/clip_by_value:z:0,tf.compat.v1.floor_div_6/FloorDiv/y:output:0*
T0*'
_output_shapes
:?????????2#
!tf.compat.v1.floor_div_6/FloorDiv?
$tf.math.greater_equal_9/GreaterEqualGreaterEqual"flatten_6/PartitionedCall:output:0&tf_math_greater_equal_9_greaterequal_y*
T0*'
_output_shapes
:?????????2&
$tf.math.greater_equal_9/GreaterEqual?
tf.math.floormod_6/FloorMod/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2
tf.math.floormod_6/FloorMod/y?
tf.math.floormod_6/FloorModFloorMod$tf.clip_by_value_9/clip_by_value:z:0&tf.math.floormod_6/FloorMod/y:output:0*
T0*'
_output_shapes
:?????????2
tf.math.floormod_6/FloorMod?
$embedding_20/StatefulPartitionedCallStatefulPartitionedCall$tf.clip_by_value_9/clip_by_value:z:0embedding_20_20517488*
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
J__inference_embedding_20_layer_call_and_return_conditional_losses_205174092&
$embedding_20/StatefulPartitionedCall?
$embedding_18/StatefulPartitionedCallStatefulPartitionedCall%tf.compat.v1.floor_div_6/FloorDiv:z:0embedding_18_20517491*
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
J__inference_embedding_18_layer_call_and_return_conditional_losses_205174312&
$embedding_18/StatefulPartitionedCall?
tf.cast_9/CastCast(tf.math.greater_equal_9/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2
tf.cast_9/Cast?
tf.__operators__.add_18/AddV2AddV2-embedding_20/StatefulPartitionedCall:output:0-embedding_18/StatefulPartitionedCall:output:0*
T0*,
_output_shapes
:??????????2
tf.__operators__.add_18/AddV2?
$embedding_19/StatefulPartitionedCallStatefulPartitionedCalltf.math.floormod_6/FloorMod:z:0embedding_19_20517496*
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
J__inference_embedding_19_layer_call_and_return_conditional_losses_205174552&
$embedding_19/StatefulPartitionedCall?
tf.__operators__.add_19/AddV2AddV2!tf.__operators__.add_18/AddV2:z:0-embedding_19/StatefulPartitionedCall:output:0*
T0*,
_output_shapes
:??????????2
tf.__operators__.add_19/AddV2?
tf.expand_dims_6/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2!
tf.expand_dims_6/ExpandDims/dim?
tf.expand_dims_6/ExpandDims
ExpandDimstf.cast_9/Cast:y:0(tf.expand_dims_6/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????2
tf.expand_dims_6/ExpandDims?
tf.math.multiply_6/MulMul!tf.__operators__.add_19/AddV2:z:0$tf.expand_dims_6/ExpandDims:output:0*
T0*,
_output_shapes
:??????????2
tf.math.multiply_6/Mul?
*tf.math.reduce_sum_6/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2,
*tf.math.reduce_sum_6/Sum/reduction_indices?
tf.math.reduce_sum_6/SumSumtf.math.multiply_6/Mul:z:03tf.math.reduce_sum_6/Sum/reduction_indices:output:0*
T0*(
_output_shapes
:??????????2
tf.math.reduce_sum_6/Sum?
IdentityIdentity!tf.math.reduce_sum_6/Sum:output:0%^embedding_18/StatefulPartitionedCall%^embedding_19/StatefulPartitionedCall%^embedding_20/StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:?????????: :::2L
$embedding_18/StatefulPartitionedCall$embedding_18/StatefulPartitionedCall2L
$embedding_19/StatefulPartitionedCall$embedding_19/StatefulPartitionedCall2L
$embedding_20/StatefulPartitionedCall$embedding_20/StatefulPartitionedCall:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_7:

_output_shapes
: 
?
u
/__inference_embedding_20_layer_call_fn_20519608

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
J__inference_embedding_20_layer_call_and_return_conditional_losses_205174092
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
u
/__inference_embedding_22_layer_call_fn_20519704

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
J__inference_embedding_22_layer_call_and_return_conditional_losses_205176812
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
??
?,
$__inference__traced_restore_20520236
file_prefix$
 assignvariableop_dense_27_kernel$
 assignvariableop_1_dense_27_bias&
"assignvariableop_2_dense_28_kernel$
 assignvariableop_3_dense_28_bias&
"assignvariableop_4_dense_30_kernel$
 assignvariableop_5_dense_30_bias&
"assignvariableop_6_dense_29_kernel$
 assignvariableop_7_dense_29_bias&
"assignvariableop_8_dense_31_kernel$
 assignvariableop_9_dense_31_bias'
#assignvariableop_10_dense_32_kernel%
!assignvariableop_11_dense_32_bias'
#assignvariableop_12_dense_33_kernel%
!assignvariableop_13_dense_33_bias'
#assignvariableop_14_dense_34_kernel%
!assignvariableop_15_dense_34_bias'
#assignvariableop_16_dense_35_kernel%
!assignvariableop_17_dense_35_bias!
assignvariableop_18_adam_iter#
assignvariableop_19_adam_beta_1#
assignvariableop_20_adam_beta_2"
assignvariableop_21_adam_decay*
&assignvariableop_22_adam_learning_rate/
+assignvariableop_23_embedding_20_embeddings/
+assignvariableop_24_embedding_18_embeddings/
+assignvariableop_25_embedding_19_embeddings/
+assignvariableop_26_embedding_23_embeddings/
+assignvariableop_27_embedding_21_embeddings/
+assignvariableop_28_embedding_22_embeddings8
4assignvariableop_29_normalize_3_normalization_3_mean<
8assignvariableop_30_normalize_3_normalization_3_variance9
5assignvariableop_31_normalize_3_normalization_3_count
assignvariableop_32_total
assignvariableop_33_count.
*assignvariableop_34_adam_dense_27_kernel_m,
(assignvariableop_35_adam_dense_27_bias_m.
*assignvariableop_36_adam_dense_28_kernel_m,
(assignvariableop_37_adam_dense_28_bias_m.
*assignvariableop_38_adam_dense_30_kernel_m,
(assignvariableop_39_adam_dense_30_bias_m.
*assignvariableop_40_adam_dense_29_kernel_m,
(assignvariableop_41_adam_dense_29_bias_m.
*assignvariableop_42_adam_dense_31_kernel_m,
(assignvariableop_43_adam_dense_31_bias_m.
*assignvariableop_44_adam_dense_32_kernel_m,
(assignvariableop_45_adam_dense_32_bias_m.
*assignvariableop_46_adam_dense_33_kernel_m,
(assignvariableop_47_adam_dense_33_bias_m.
*assignvariableop_48_adam_dense_34_kernel_m,
(assignvariableop_49_adam_dense_34_bias_m.
*assignvariableop_50_adam_dense_35_kernel_m,
(assignvariableop_51_adam_dense_35_bias_m6
2assignvariableop_52_adam_embedding_20_embeddings_m6
2assignvariableop_53_adam_embedding_18_embeddings_m6
2assignvariableop_54_adam_embedding_19_embeddings_m6
2assignvariableop_55_adam_embedding_23_embeddings_m6
2assignvariableop_56_adam_embedding_21_embeddings_m6
2assignvariableop_57_adam_embedding_22_embeddings_m.
*assignvariableop_58_adam_dense_27_kernel_v,
(assignvariableop_59_adam_dense_27_bias_v.
*assignvariableop_60_adam_dense_28_kernel_v,
(assignvariableop_61_adam_dense_28_bias_v.
*assignvariableop_62_adam_dense_30_kernel_v,
(assignvariableop_63_adam_dense_30_bias_v.
*assignvariableop_64_adam_dense_29_kernel_v,
(assignvariableop_65_adam_dense_29_bias_v.
*assignvariableop_66_adam_dense_31_kernel_v,
(assignvariableop_67_adam_dense_31_bias_v.
*assignvariableop_68_adam_dense_32_kernel_v,
(assignvariableop_69_adam_dense_32_bias_v.
*assignvariableop_70_adam_dense_33_kernel_v,
(assignvariableop_71_adam_dense_33_bias_v.
*assignvariableop_72_adam_dense_34_kernel_v,
(assignvariableop_73_adam_dense_34_bias_v.
*assignvariableop_74_adam_dense_35_kernel_v,
(assignvariableop_75_adam_dense_35_bias_v6
2assignvariableop_76_adam_embedding_20_embeddings_v6
2assignvariableop_77_adam_embedding_18_embeddings_v6
2assignvariableop_78_adam_embedding_19_embeddings_v6
2assignvariableop_79_adam_embedding_23_embeddings_v6
2assignvariableop_80_adam_embedding_21_embeddings_v6
2assignvariableop_81_adam_embedding_22_embeddings_v
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
AssignVariableOpAssignVariableOp assignvariableop_dense_27_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp assignvariableop_1_dense_27_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp"assignvariableop_2_dense_28_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp assignvariableop_3_dense_28_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp"assignvariableop_4_dense_30_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp assignvariableop_5_dense_30_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp"assignvariableop_6_dense_29_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp assignvariableop_7_dense_29_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp"assignvariableop_8_dense_31_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp assignvariableop_9_dense_31_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp#assignvariableop_10_dense_32_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOp!assignvariableop_11_dense_32_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOp#assignvariableop_12_dense_33_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOp!assignvariableop_13_dense_33_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOp#assignvariableop_14_dense_34_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOp!assignvariableop_15_dense_34_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOp#assignvariableop_16_dense_35_kernelIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOp!assignvariableop_17_dense_35_biasIdentity_17:output:0"/device:CPU:0*
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
AssignVariableOp_23AssignVariableOp+assignvariableop_23_embedding_20_embeddingsIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOp+assignvariableop_24_embedding_18_embeddingsIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOp+assignvariableop_25_embedding_19_embeddingsIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOp+assignvariableop_26_embedding_23_embeddingsIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOp+assignvariableop_27_embedding_21_embeddingsIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOp+assignvariableop_28_embedding_22_embeddingsIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOp4assignvariableop_29_normalize_3_normalization_3_meanIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOp8assignvariableop_30_normalize_3_normalization_3_varianceIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_31?
AssignVariableOp_31AssignVariableOp5assignvariableop_31_normalize_3_normalization_3_countIdentity_31:output:0"/device:CPU:0*
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
AssignVariableOp_34AssignVariableOp*assignvariableop_34_adam_dense_27_kernel_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35?
AssignVariableOp_35AssignVariableOp(assignvariableop_35_adam_dense_27_bias_mIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36?
AssignVariableOp_36AssignVariableOp*assignvariableop_36_adam_dense_28_kernel_mIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37?
AssignVariableOp_37AssignVariableOp(assignvariableop_37_adam_dense_28_bias_mIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38?
AssignVariableOp_38AssignVariableOp*assignvariableop_38_adam_dense_30_kernel_mIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39?
AssignVariableOp_39AssignVariableOp(assignvariableop_39_adam_dense_30_bias_mIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40?
AssignVariableOp_40AssignVariableOp*assignvariableop_40_adam_dense_29_kernel_mIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41?
AssignVariableOp_41AssignVariableOp(assignvariableop_41_adam_dense_29_bias_mIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42?
AssignVariableOp_42AssignVariableOp*assignvariableop_42_adam_dense_31_kernel_mIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43?
AssignVariableOp_43AssignVariableOp(assignvariableop_43_adam_dense_31_bias_mIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44?
AssignVariableOp_44AssignVariableOp*assignvariableop_44_adam_dense_32_kernel_mIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45?
AssignVariableOp_45AssignVariableOp(assignvariableop_45_adam_dense_32_bias_mIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46?
AssignVariableOp_46AssignVariableOp*assignvariableop_46_adam_dense_33_kernel_mIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47?
AssignVariableOp_47AssignVariableOp(assignvariableop_47_adam_dense_33_bias_mIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48?
AssignVariableOp_48AssignVariableOp*assignvariableop_48_adam_dense_34_kernel_mIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49?
AssignVariableOp_49AssignVariableOp(assignvariableop_49_adam_dense_34_bias_mIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50?
AssignVariableOp_50AssignVariableOp*assignvariableop_50_adam_dense_35_kernel_mIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51?
AssignVariableOp_51AssignVariableOp(assignvariableop_51_adam_dense_35_bias_mIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52?
AssignVariableOp_52AssignVariableOp2assignvariableop_52_adam_embedding_20_embeddings_mIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_52n
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:2
Identity_53?
AssignVariableOp_53AssignVariableOp2assignvariableop_53_adam_embedding_18_embeddings_mIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_53n
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:2
Identity_54?
AssignVariableOp_54AssignVariableOp2assignvariableop_54_adam_embedding_19_embeddings_mIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_54n
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:2
Identity_55?
AssignVariableOp_55AssignVariableOp2assignvariableop_55_adam_embedding_23_embeddings_mIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_55n
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:2
Identity_56?
AssignVariableOp_56AssignVariableOp2assignvariableop_56_adam_embedding_21_embeddings_mIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_56n
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:2
Identity_57?
AssignVariableOp_57AssignVariableOp2assignvariableop_57_adam_embedding_22_embeddings_mIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_57n
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:2
Identity_58?
AssignVariableOp_58AssignVariableOp*assignvariableop_58_adam_dense_27_kernel_vIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_58n
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:2
Identity_59?
AssignVariableOp_59AssignVariableOp(assignvariableop_59_adam_dense_27_bias_vIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_59n
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:2
Identity_60?
AssignVariableOp_60AssignVariableOp*assignvariableop_60_adam_dense_28_kernel_vIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_60n
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:2
Identity_61?
AssignVariableOp_61AssignVariableOp(assignvariableop_61_adam_dense_28_bias_vIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_61n
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:2
Identity_62?
AssignVariableOp_62AssignVariableOp*assignvariableop_62_adam_dense_30_kernel_vIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_62n
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:2
Identity_63?
AssignVariableOp_63AssignVariableOp(assignvariableop_63_adam_dense_30_bias_vIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_63n
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:2
Identity_64?
AssignVariableOp_64AssignVariableOp*assignvariableop_64_adam_dense_29_kernel_vIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_64n
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:2
Identity_65?
AssignVariableOp_65AssignVariableOp(assignvariableop_65_adam_dense_29_bias_vIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_65n
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:2
Identity_66?
AssignVariableOp_66AssignVariableOp*assignvariableop_66_adam_dense_31_kernel_vIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_66n
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:2
Identity_67?
AssignVariableOp_67AssignVariableOp(assignvariableop_67_adam_dense_31_bias_vIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_67n
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:2
Identity_68?
AssignVariableOp_68AssignVariableOp*assignvariableop_68_adam_dense_32_kernel_vIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_68n
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:2
Identity_69?
AssignVariableOp_69AssignVariableOp(assignvariableop_69_adam_dense_32_bias_vIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_69n
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:2
Identity_70?
AssignVariableOp_70AssignVariableOp*assignvariableop_70_adam_dense_33_kernel_vIdentity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_70n
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:2
Identity_71?
AssignVariableOp_71AssignVariableOp(assignvariableop_71_adam_dense_33_bias_vIdentity_71:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_71n
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:2
Identity_72?
AssignVariableOp_72AssignVariableOp*assignvariableop_72_adam_dense_34_kernel_vIdentity_72:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_72n
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:2
Identity_73?
AssignVariableOp_73AssignVariableOp(assignvariableop_73_adam_dense_34_bias_vIdentity_73:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_73n
Identity_74IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:2
Identity_74?
AssignVariableOp_74AssignVariableOp*assignvariableop_74_adam_dense_35_kernel_vIdentity_74:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_74n
Identity_75IdentityRestoreV2:tensors:75"/device:CPU:0*
T0*
_output_shapes
:2
Identity_75?
AssignVariableOp_75AssignVariableOp(assignvariableop_75_adam_dense_35_bias_vIdentity_75:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_75n
Identity_76IdentityRestoreV2:tensors:76"/device:CPU:0*
T0*
_output_shapes
:2
Identity_76?
AssignVariableOp_76AssignVariableOp2assignvariableop_76_adam_embedding_20_embeddings_vIdentity_76:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_76n
Identity_77IdentityRestoreV2:tensors:77"/device:CPU:0*
T0*
_output_shapes
:2
Identity_77?
AssignVariableOp_77AssignVariableOp2assignvariableop_77_adam_embedding_18_embeddings_vIdentity_77:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_77n
Identity_78IdentityRestoreV2:tensors:78"/device:CPU:0*
T0*
_output_shapes
:2
Identity_78?
AssignVariableOp_78AssignVariableOp2assignvariableop_78_adam_embedding_19_embeddings_vIdentity_78:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_78n
Identity_79IdentityRestoreV2:tensors:79"/device:CPU:0*
T0*
_output_shapes
:2
Identity_79?
AssignVariableOp_79AssignVariableOp2assignvariableop_79_adam_embedding_23_embeddings_vIdentity_79:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_79n
Identity_80IdentityRestoreV2:tensors:80"/device:CPU:0*
T0*
_output_shapes
:2
Identity_80?
AssignVariableOp_80AssignVariableOp2assignvariableop_80_adam_embedding_21_embeddings_vIdentity_80:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_80n
Identity_81IdentityRestoreV2:tensors:81"/device:CPU:0*
T0*
_output_shapes
:2
Identity_81?
AssignVariableOp_81AssignVariableOp2assignvariableop_81_adam_embedding_22_embeddings_vIdentity_81:output:0"/device:CPU:0*
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
J__inference_embedding_18_layer_call_and_return_conditional_losses_20519618

inputs
embedding_lookup_20519612
identity??embedding_lookup]
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:?????????2
Cast?
embedding_lookupResourceGatherembedding_lookup_20519612Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*,
_class"
 loc:@embedding_lookup/20519612*,
_output_shapes
:??????????*
dtype02
embedding_lookup?
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*,
_class"
 loc:@embedding_lookup/20519612*,
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
c
G__inference_flatten_7_layer_call_and_return_conditional_losses_20519648

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
?-
?
E__inference_model_7_layer_call_and_return_conditional_losses_20517812

inputs+
'tf_math_greater_equal_10_greaterequal_y
embedding_23_20517794
embedding_21_20517797
embedding_22_20517802
identity??$embedding_21/StatefulPartitionedCall?$embedding_22/StatefulPartitionedCall?$embedding_23/StatefulPartitionedCall?
flatten_7/PartitionedCallPartitionedCallinputs*
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
GPU2 *0J 8? *P
fKRI
G__inference_flatten_7_layer_call_and_return_conditional_losses_205176072
flatten_7/PartitionedCall?
+tf.clip_by_value_10/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 * $tI2-
+tf.clip_by_value_10/clip_by_value/Minimum/y?
)tf.clip_by_value_10/clip_by_value/MinimumMinimum"flatten_7/PartitionedCall:output:04tf.clip_by_value_10/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2+
)tf.clip_by_value_10/clip_by_value/Minimum?
#tf.clip_by_value_10/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2%
#tf.clip_by_value_10/clip_by_value/y?
!tf.clip_by_value_10/clip_by_valueMaximum-tf.clip_by_value_10/clip_by_value/Minimum:z:0,tf.clip_by_value_10/clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????2#
!tf.clip_by_value_10/clip_by_value?
#tf.compat.v1.floor_div_7/FloorDiv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2%
#tf.compat.v1.floor_div_7/FloorDiv/y?
!tf.compat.v1.floor_div_7/FloorDivFloorDiv%tf.clip_by_value_10/clip_by_value:z:0,tf.compat.v1.floor_div_7/FloorDiv/y:output:0*
T0*'
_output_shapes
:?????????2#
!tf.compat.v1.floor_div_7/FloorDiv?
%tf.math.greater_equal_10/GreaterEqualGreaterEqual"flatten_7/PartitionedCall:output:0'tf_math_greater_equal_10_greaterequal_y*
T0*'
_output_shapes
:?????????2'
%tf.math.greater_equal_10/GreaterEqual?
tf.math.floormod_7/FloorMod/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2
tf.math.floormod_7/FloorMod/y?
tf.math.floormod_7/FloorModFloorMod%tf.clip_by_value_10/clip_by_value:z:0&tf.math.floormod_7/FloorMod/y:output:0*
T0*'
_output_shapes
:?????????2
tf.math.floormod_7/FloorMod?
$embedding_23/StatefulPartitionedCallStatefulPartitionedCall%tf.clip_by_value_10/clip_by_value:z:0embedding_23_20517794*
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
J__inference_embedding_23_layer_call_and_return_conditional_losses_205176352&
$embedding_23/StatefulPartitionedCall?
$embedding_21/StatefulPartitionedCallStatefulPartitionedCall%tf.compat.v1.floor_div_7/FloorDiv:z:0embedding_21_20517797*
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
J__inference_embedding_21_layer_call_and_return_conditional_losses_205176572&
$embedding_21/StatefulPartitionedCall?
tf.cast_10/CastCast)tf.math.greater_equal_10/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2
tf.cast_10/Cast?
tf.__operators__.add_20/AddV2AddV2-embedding_23/StatefulPartitionedCall:output:0-embedding_21/StatefulPartitionedCall:output:0*
T0*,
_output_shapes
:??????????2
tf.__operators__.add_20/AddV2?
$embedding_22/StatefulPartitionedCallStatefulPartitionedCalltf.math.floormod_7/FloorMod:z:0embedding_22_20517802*
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
J__inference_embedding_22_layer_call_and_return_conditional_losses_205176812&
$embedding_22/StatefulPartitionedCall?
tf.__operators__.add_21/AddV2AddV2!tf.__operators__.add_20/AddV2:z:0-embedding_22/StatefulPartitionedCall:output:0*
T0*,
_output_shapes
:??????????2
tf.__operators__.add_21/AddV2?
tf.expand_dims_7/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2!
tf.expand_dims_7/ExpandDims/dim?
tf.expand_dims_7/ExpandDims
ExpandDimstf.cast_10/Cast:y:0(tf.expand_dims_7/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????2
tf.expand_dims_7/ExpandDims?
tf.math.multiply_7/MulMul!tf.__operators__.add_21/AddV2:z:0$tf.expand_dims_7/ExpandDims:output:0*
T0*,
_output_shapes
:??????????2
tf.math.multiply_7/Mul?
*tf.math.reduce_sum_7/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2,
*tf.math.reduce_sum_7/Sum/reduction_indices?
tf.math.reduce_sum_7/SumSumtf.math.multiply_7/Mul:z:03tf.math.reduce_sum_7/Sum/reduction_indices:output:0*
T0*(
_output_shapes
:??????????2
tf.math.reduce_sum_7/Sum?
IdentityIdentity!tf.math.reduce_sum_7/Sum:output:0%^embedding_21/StatefulPartitionedCall%^embedding_22/StatefulPartitionedCall%^embedding_23/StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:?????????: :::2L
$embedding_21/StatefulPartitionedCall$embedding_21/StatefulPartitionedCall2L
$embedding_22/StatefulPartitionedCall$embedding_22/StatefulPartitionedCall2L
$embedding_23/StatefulPartitionedCall$embedding_23/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: 
?
?
*__inference_model_6_layer_call_fn_20519257

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
GPU2 *0J 8? *N
fIRG
E__inference_model_6_layer_call_and_return_conditional_losses_205175412
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
?	
?
F__inference_dense_29_layer_call_and_return_conditional_losses_20519450

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
?	
?
F__inference_dense_28_layer_call_and_return_conditional_losses_20517974

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
?
+__inference_dense_35_layer_call_fn_20519580

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
F__inference_dense_35_layer_call_and_return_conditional_losses_205181712
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
?
?
+__inference_dense_33_layer_call_fn_20519516

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
F__inference_dense_33_layer_call_and_return_conditional_losses_205180822
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
J__inference_embedding_19_layer_call_and_return_conditional_losses_20519635

inputs
embedding_lookup_20519629
identity??embedding_lookup]
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:?????????2
Cast?
embedding_lookupResourceGatherembedding_lookup_20519629Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*,
_class"
 loc:@embedding_lookup/20519629*,
_output_shapes
:??????????*
dtype02
embedding_lookup?
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*,
_class"
 loc:@embedding_lookup/20519629*,
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
*__inference_model_7_layer_call_fn_20517823
input_8
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_8unknown	unknown_0	unknown_1	unknown_2*
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
GPU2 *0J 8? *N
fIRG
E__inference_model_7_layer_call_and_return_conditional_losses_205178122
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:?????????: :::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_8:

_output_shapes
: 
?
H
,__inference_flatten_7_layer_call_fn_20519653

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
GPU2 *0J 8? *P
fKRI
G__inference_flatten_7_layer_call_and_return_conditional_losses_205176072
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
?X
?

L__inference_custom_model_3_layer_call_and_return_conditional_losses_20518188

cards0

cards1
bets+
'tf_math_greater_equal_11_greaterequal_y
model_6_20517857
model_6_20517859
model_6_20517861
model_6_20517863
model_7_20517892
model_7_20517894
model_7_20517896
model_7_20517898/
+tf_clip_by_value_11_clip_by_value_minimum_y'
#tf_clip_by_value_11_clip_by_value_y
dense_27_20517932
dense_27_20517934
dense_30_20517958
dense_30_20517960
dense_28_20517985
dense_28_20517987
dense_29_20518012
dense_29_20518014
dense_31_20518038
dense_31_20518040
dense_32_20518066
dense_32_20518068
dense_33_20518093
dense_33_20518095
dense_34_20518121
dense_34_20518123
normalize_3_20518156
normalize_3_20518158
dense_35_20518182
dense_35_20518184
identity?? dense_27/StatefulPartitionedCall? dense_28/StatefulPartitionedCall? dense_29/StatefulPartitionedCall? dense_30/StatefulPartitionedCall? dense_31/StatefulPartitionedCall? dense_32/StatefulPartitionedCall? dense_33/StatefulPartitionedCall? dense_34/StatefulPartitionedCall? dense_35/StatefulPartitionedCall?model_6/StatefulPartitionedCall?model_7/StatefulPartitionedCall?#normalize_3/StatefulPartitionedCall?
%tf.math.greater_equal_11/GreaterEqualGreaterEqualbets'tf_math_greater_equal_11_greaterequal_y*
T0*'
_output_shapes
:?????????
2'
%tf.math.greater_equal_11/GreaterEqual?
model_6/StatefulPartitionedCallStatefulPartitionedCallcards0model_6_20517857model_6_20517859model_6_20517861model_6_20517863*
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
GPU2 *0J 8? *N
fIRG
E__inference_model_6_layer_call_and_return_conditional_losses_205175412!
model_6/StatefulPartitionedCall?
model_7/StatefulPartitionedCallStatefulPartitionedCallcards1model_7_20517892model_7_20517894model_7_20517896model_7_20517898*
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
GPU2 *0J 8? *N
fIRG
E__inference_model_7_layer_call_and_return_conditional_losses_205177672!
model_7/StatefulPartitionedCall?
)tf.clip_by_value_11/clip_by_value/MinimumMinimumbets+tf_clip_by_value_11_clip_by_value_minimum_y*
T0*'
_output_shapes
:?????????
2+
)tf.clip_by_value_11/clip_by_value/Minimum?
!tf.clip_by_value_11/clip_by_valueMaximum-tf.clip_by_value_11/clip_by_value/Minimum:z:0#tf_clip_by_value_11_clip_by_value_y*
T0*'
_output_shapes
:?????????
2#
!tf.clip_by_value_11/clip_by_value?
tf.cast_11/CastCast)tf.math.greater_equal_11/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????
2
tf.cast_11/Castt
tf.concat_9/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
tf.concat_9/concat/axis?
tf.concat_9/concatConcatV2(model_6/StatefulPartitionedCall:output:0(model_7/StatefulPartitionedCall:output:0 tf.concat_9/concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????2
tf.concat_9/concat
tf.concat_10/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
tf.concat_10/concat/axis?
tf.concat_10/concatConcatV2%tf.clip_by_value_11/clip_by_value:z:0tf.cast_11/Cast:y:0!tf.concat_10/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????2
tf.concat_10/concat?
 dense_27/StatefulPartitionedCallStatefulPartitionedCalltf.concat_9/concat:output:0dense_27_20517932dense_27_20517934*
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
F__inference_dense_27_layer_call_and_return_conditional_losses_205179212"
 dense_27/StatefulPartitionedCall?
 dense_30/StatefulPartitionedCallStatefulPartitionedCalltf.concat_10/concat:output:0dense_30_20517958dense_30_20517960*
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
F__inference_dense_30_layer_call_and_return_conditional_losses_205179472"
 dense_30/StatefulPartitionedCall?
 dense_28/StatefulPartitionedCallStatefulPartitionedCall)dense_27/StatefulPartitionedCall:output:0dense_28_20517985dense_28_20517987*
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
F__inference_dense_28_layer_call_and_return_conditional_losses_205179742"
 dense_28/StatefulPartitionedCall?
 dense_29/StatefulPartitionedCallStatefulPartitionedCall)dense_28/StatefulPartitionedCall:output:0dense_29_20518012dense_29_20518014*
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
F__inference_dense_29_layer_call_and_return_conditional_losses_205180012"
 dense_29/StatefulPartitionedCall?
 dense_31/StatefulPartitionedCallStatefulPartitionedCall)dense_30/StatefulPartitionedCall:output:0dense_31_20518038dense_31_20518040*
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
F__inference_dense_31_layer_call_and_return_conditional_losses_205180272"
 dense_31/StatefulPartitionedCall
tf.concat_11/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
tf.concat_11/concat/axis?
tf.concat_11/concatConcatV2)dense_29/StatefulPartitionedCall:output:0)dense_31/StatefulPartitionedCall:output:0!tf.concat_11/concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????2
tf.concat_11/concat?
 dense_32/StatefulPartitionedCallStatefulPartitionedCalltf.concat_11/concat:output:0dense_32_20518066dense_32_20518068*
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
F__inference_dense_32_layer_call_and_return_conditional_losses_205180552"
 dense_32/StatefulPartitionedCall?
tf.nn.relu_9/ReluRelu)dense_32/StatefulPartitionedCall:output:0*
T0*(
_output_shapes
:??????????2
tf.nn.relu_9/Relu?
 dense_33/StatefulPartitionedCallStatefulPartitionedCalltf.nn.relu_9/Relu:activations:0dense_33_20518093dense_33_20518095*
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
F__inference_dense_33_layer_call_and_return_conditional_losses_205180822"
 dense_33/StatefulPartitionedCall?
tf.__operators__.add_22/AddV2AddV2)dense_33/StatefulPartitionedCall:output:0tf.nn.relu_9/Relu:activations:0*
T0*(
_output_shapes
:??????????2
tf.__operators__.add_22/AddV2?
tf.nn.relu_10/ReluRelu!tf.__operators__.add_22/AddV2:z:0*
T0*(
_output_shapes
:??????????2
tf.nn.relu_10/Relu?
 dense_34/StatefulPartitionedCallStatefulPartitionedCall tf.nn.relu_10/Relu:activations:0dense_34_20518121dense_34_20518123*
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
F__inference_dense_34_layer_call_and_return_conditional_losses_205181102"
 dense_34/StatefulPartitionedCall?
tf.__operators__.add_23/AddV2AddV2)dense_34/StatefulPartitionedCall:output:0 tf.nn.relu_10/Relu:activations:0*
T0*(
_output_shapes
:??????????2
tf.__operators__.add_23/AddV2?
tf.nn.relu_11/ReluRelu!tf.__operators__.add_23/AddV2:z:0*
T0*(
_output_shapes
:??????????2
tf.nn.relu_11/Relu?
#normalize_3/StatefulPartitionedCallStatefulPartitionedCall tf.nn.relu_11/Relu:activations:0normalize_3_20518156normalize_3_20518158*
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
I__inference_normalize_3_layer_call_and_return_conditional_losses_205181452%
#normalize_3/StatefulPartitionedCall?
 dense_35/StatefulPartitionedCallStatefulPartitionedCall,normalize_3/StatefulPartitionedCall:output:0dense_35_20518182dense_35_20518184*
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
F__inference_dense_35_layer_call_and_return_conditional_losses_205181712"
 dense_35/StatefulPartitionedCall?
IdentityIdentity)dense_35/StatefulPartitionedCall:output:0!^dense_27/StatefulPartitionedCall!^dense_28/StatefulPartitionedCall!^dense_29/StatefulPartitionedCall!^dense_30/StatefulPartitionedCall!^dense_31/StatefulPartitionedCall!^dense_32/StatefulPartitionedCall!^dense_33/StatefulPartitionedCall!^dense_34/StatefulPartitionedCall!^dense_35/StatefulPartitionedCall ^model_6/StatefulPartitionedCall ^model_7/StatefulPartitionedCall$^normalize_3/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????:?????????:?????????
: : :::: :::: : ::::::::::::::::::::2D
 dense_27/StatefulPartitionedCall dense_27/StatefulPartitionedCall2D
 dense_28/StatefulPartitionedCall dense_28/StatefulPartitionedCall2D
 dense_29/StatefulPartitionedCall dense_29/StatefulPartitionedCall2D
 dense_30/StatefulPartitionedCall dense_30/StatefulPartitionedCall2D
 dense_31/StatefulPartitionedCall dense_31/StatefulPartitionedCall2D
 dense_32/StatefulPartitionedCall dense_32/StatefulPartitionedCall2D
 dense_33/StatefulPartitionedCall dense_33/StatefulPartitionedCall2D
 dense_34/StatefulPartitionedCall dense_34/StatefulPartitionedCall2D
 dense_35/StatefulPartitionedCall dense_35/StatefulPartitionedCall2B
model_6/StatefulPartitionedCallmodel_6/StatefulPartitionedCall2B
model_7/StatefulPartitionedCallmodel_7/StatefulPartitionedCall2J
#normalize_3/StatefulPartitionedCall#normalize_3/StatefulPartitionedCall:O K
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
?-
?
E__inference_model_6_layer_call_and_return_conditional_losses_20517586

inputs*
&tf_math_greater_equal_9_greaterequal_y
embedding_20_20517568
embedding_18_20517571
embedding_19_20517576
identity??$embedding_18/StatefulPartitionedCall?$embedding_19/StatefulPartitionedCall?$embedding_20/StatefulPartitionedCall?
flatten_6/PartitionedCallPartitionedCallinputs*
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
GPU2 *0J 8? *P
fKRI
G__inference_flatten_6_layer_call_and_return_conditional_losses_205173812
flatten_6/PartitionedCall?
*tf.clip_by_value_9/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 * $tI2,
*tf.clip_by_value_9/clip_by_value/Minimum/y?
(tf.clip_by_value_9/clip_by_value/MinimumMinimum"flatten_6/PartitionedCall:output:03tf.clip_by_value_9/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2*
(tf.clip_by_value_9/clip_by_value/Minimum?
"tf.clip_by_value_9/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"tf.clip_by_value_9/clip_by_value/y?
 tf.clip_by_value_9/clip_by_valueMaximum,tf.clip_by_value_9/clip_by_value/Minimum:z:0+tf.clip_by_value_9/clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????2"
 tf.clip_by_value_9/clip_by_value?
#tf.compat.v1.floor_div_6/FloorDiv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2%
#tf.compat.v1.floor_div_6/FloorDiv/y?
!tf.compat.v1.floor_div_6/FloorDivFloorDiv$tf.clip_by_value_9/clip_by_value:z:0,tf.compat.v1.floor_div_6/FloorDiv/y:output:0*
T0*'
_output_shapes
:?????????2#
!tf.compat.v1.floor_div_6/FloorDiv?
$tf.math.greater_equal_9/GreaterEqualGreaterEqual"flatten_6/PartitionedCall:output:0&tf_math_greater_equal_9_greaterequal_y*
T0*'
_output_shapes
:?????????2&
$tf.math.greater_equal_9/GreaterEqual?
tf.math.floormod_6/FloorMod/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2
tf.math.floormod_6/FloorMod/y?
tf.math.floormod_6/FloorModFloorMod$tf.clip_by_value_9/clip_by_value:z:0&tf.math.floormod_6/FloorMod/y:output:0*
T0*'
_output_shapes
:?????????2
tf.math.floormod_6/FloorMod?
$embedding_20/StatefulPartitionedCallStatefulPartitionedCall$tf.clip_by_value_9/clip_by_value:z:0embedding_20_20517568*
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
J__inference_embedding_20_layer_call_and_return_conditional_losses_205174092&
$embedding_20/StatefulPartitionedCall?
$embedding_18/StatefulPartitionedCallStatefulPartitionedCall%tf.compat.v1.floor_div_6/FloorDiv:z:0embedding_18_20517571*
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
J__inference_embedding_18_layer_call_and_return_conditional_losses_205174312&
$embedding_18/StatefulPartitionedCall?
tf.cast_9/CastCast(tf.math.greater_equal_9/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2
tf.cast_9/Cast?
tf.__operators__.add_18/AddV2AddV2-embedding_20/StatefulPartitionedCall:output:0-embedding_18/StatefulPartitionedCall:output:0*
T0*,
_output_shapes
:??????????2
tf.__operators__.add_18/AddV2?
$embedding_19/StatefulPartitionedCallStatefulPartitionedCalltf.math.floormod_6/FloorMod:z:0embedding_19_20517576*
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
J__inference_embedding_19_layer_call_and_return_conditional_losses_205174552&
$embedding_19/StatefulPartitionedCall?
tf.__operators__.add_19/AddV2AddV2!tf.__operators__.add_18/AddV2:z:0-embedding_19/StatefulPartitionedCall:output:0*
T0*,
_output_shapes
:??????????2
tf.__operators__.add_19/AddV2?
tf.expand_dims_6/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2!
tf.expand_dims_6/ExpandDims/dim?
tf.expand_dims_6/ExpandDims
ExpandDimstf.cast_9/Cast:y:0(tf.expand_dims_6/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????2
tf.expand_dims_6/ExpandDims?
tf.math.multiply_6/MulMul!tf.__operators__.add_19/AddV2:z:0$tf.expand_dims_6/ExpandDims:output:0*
T0*,
_output_shapes
:??????????2
tf.math.multiply_6/Mul?
*tf.math.reduce_sum_6/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2,
*tf.math.reduce_sum_6/Sum/reduction_indices?
tf.math.reduce_sum_6/SumSumtf.math.multiply_6/Mul:z:03tf.math.reduce_sum_6/Sum/reduction_indices:output:0*
T0*(
_output_shapes
:??????????2
tf.math.reduce_sum_6/Sum?
IdentityIdentity!tf.math.reduce_sum_6/Sum:output:0%^embedding_18/StatefulPartitionedCall%^embedding_19/StatefulPartitionedCall%^embedding_20/StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:?????????: :::2L
$embedding_18/StatefulPartitionedCall$embedding_18/StatefulPartitionedCall2L
$embedding_19/StatefulPartitionedCall$embedding_19/StatefulPartitionedCall2L
$embedding_20/StatefulPartitionedCall$embedding_20/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: 
?-
?
E__inference_model_7_layer_call_and_return_conditional_losses_20517767

inputs+
'tf_math_greater_equal_10_greaterequal_y
embedding_23_20517749
embedding_21_20517752
embedding_22_20517757
identity??$embedding_21/StatefulPartitionedCall?$embedding_22/StatefulPartitionedCall?$embedding_23/StatefulPartitionedCall?
flatten_7/PartitionedCallPartitionedCallinputs*
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
GPU2 *0J 8? *P
fKRI
G__inference_flatten_7_layer_call_and_return_conditional_losses_205176072
flatten_7/PartitionedCall?
+tf.clip_by_value_10/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 * $tI2-
+tf.clip_by_value_10/clip_by_value/Minimum/y?
)tf.clip_by_value_10/clip_by_value/MinimumMinimum"flatten_7/PartitionedCall:output:04tf.clip_by_value_10/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2+
)tf.clip_by_value_10/clip_by_value/Minimum?
#tf.clip_by_value_10/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2%
#tf.clip_by_value_10/clip_by_value/y?
!tf.clip_by_value_10/clip_by_valueMaximum-tf.clip_by_value_10/clip_by_value/Minimum:z:0,tf.clip_by_value_10/clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????2#
!tf.clip_by_value_10/clip_by_value?
#tf.compat.v1.floor_div_7/FloorDiv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2%
#tf.compat.v1.floor_div_7/FloorDiv/y?
!tf.compat.v1.floor_div_7/FloorDivFloorDiv%tf.clip_by_value_10/clip_by_value:z:0,tf.compat.v1.floor_div_7/FloorDiv/y:output:0*
T0*'
_output_shapes
:?????????2#
!tf.compat.v1.floor_div_7/FloorDiv?
%tf.math.greater_equal_10/GreaterEqualGreaterEqual"flatten_7/PartitionedCall:output:0'tf_math_greater_equal_10_greaterequal_y*
T0*'
_output_shapes
:?????????2'
%tf.math.greater_equal_10/GreaterEqual?
tf.math.floormod_7/FloorMod/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2
tf.math.floormod_7/FloorMod/y?
tf.math.floormod_7/FloorModFloorMod%tf.clip_by_value_10/clip_by_value:z:0&tf.math.floormod_7/FloorMod/y:output:0*
T0*'
_output_shapes
:?????????2
tf.math.floormod_7/FloorMod?
$embedding_23/StatefulPartitionedCallStatefulPartitionedCall%tf.clip_by_value_10/clip_by_value:z:0embedding_23_20517749*
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
J__inference_embedding_23_layer_call_and_return_conditional_losses_205176352&
$embedding_23/StatefulPartitionedCall?
$embedding_21/StatefulPartitionedCallStatefulPartitionedCall%tf.compat.v1.floor_div_7/FloorDiv:z:0embedding_21_20517752*
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
J__inference_embedding_21_layer_call_and_return_conditional_losses_205176572&
$embedding_21/StatefulPartitionedCall?
tf.cast_10/CastCast)tf.math.greater_equal_10/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2
tf.cast_10/Cast?
tf.__operators__.add_20/AddV2AddV2-embedding_23/StatefulPartitionedCall:output:0-embedding_21/StatefulPartitionedCall:output:0*
T0*,
_output_shapes
:??????????2
tf.__operators__.add_20/AddV2?
$embedding_22/StatefulPartitionedCallStatefulPartitionedCalltf.math.floormod_7/FloorMod:z:0embedding_22_20517757*
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
J__inference_embedding_22_layer_call_and_return_conditional_losses_205176812&
$embedding_22/StatefulPartitionedCall?
tf.__operators__.add_21/AddV2AddV2!tf.__operators__.add_20/AddV2:z:0-embedding_22/StatefulPartitionedCall:output:0*
T0*,
_output_shapes
:??????????2
tf.__operators__.add_21/AddV2?
tf.expand_dims_7/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2!
tf.expand_dims_7/ExpandDims/dim?
tf.expand_dims_7/ExpandDims
ExpandDimstf.cast_10/Cast:y:0(tf.expand_dims_7/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????2
tf.expand_dims_7/ExpandDims?
tf.math.multiply_7/MulMul!tf.__operators__.add_21/AddV2:z:0$tf.expand_dims_7/ExpandDims:output:0*
T0*,
_output_shapes
:??????????2
tf.math.multiply_7/Mul?
*tf.math.reduce_sum_7/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2,
*tf.math.reduce_sum_7/Sum/reduction_indices?
tf.math.reduce_sum_7/SumSumtf.math.multiply_7/Mul:z:03tf.math.reduce_sum_7/Sum/reduction_indices:output:0*
T0*(
_output_shapes
:??????????2
tf.math.reduce_sum_7/Sum?
IdentityIdentity!tf.math.reduce_sum_7/Sum:output:0%^embedding_21/StatefulPartitionedCall%^embedding_22/StatefulPartitionedCall%^embedding_23/StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:?????????: :::2L
$embedding_21/StatefulPartitionedCall$embedding_21/StatefulPartitionedCall2L
$embedding_22/StatefulPartitionedCall$embedding_22/StatefulPartitionedCall2L
$embedding_23/StatefulPartitionedCall$embedding_23/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: 
?	
?
F__inference_dense_29_layer_call_and_return_conditional_losses_20518001

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
?	
?
J__inference_embedding_22_layer_call_and_return_conditional_losses_20519697

inputs
embedding_lookup_20519691
identity??embedding_lookup]
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:?????????2
Cast?
embedding_lookupResourceGatherembedding_lookup_20519691Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*,
_class"
 loc:@embedding_lookup/20519691*,
_output_shapes
:??????????*
dtype02
embedding_lookup?
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*,
_class"
 loc:@embedding_lookup/20519691*,
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
?
?
*__inference_model_6_layer_call_fn_20517597
input_7
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_7unknown	unknown_0	unknown_1	unknown_2*
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
GPU2 *0J 8? *N
fIRG
E__inference_model_6_layer_call_and_return_conditional_losses_205175862
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:?????????: :::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_7:

_output_shapes
: 
?9
?
E__inference_model_7_layer_call_and_return_conditional_losses_20519354

inputs+
'tf_math_greater_equal_10_greaterequal_y*
&embedding_23_embedding_lookup_20519328*
&embedding_21_embedding_lookup_20519334*
&embedding_22_embedding_lookup_20519342
identity??embedding_21/embedding_lookup?embedding_22/embedding_lookup?embedding_23/embedding_lookups
flatten_7/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
flatten_7/Const?
flatten_7/ReshapeReshapeinputsflatten_7/Const:output:0*
T0*'
_output_shapes
:?????????2
flatten_7/Reshape?
+tf.clip_by_value_10/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 * $tI2-
+tf.clip_by_value_10/clip_by_value/Minimum/y?
)tf.clip_by_value_10/clip_by_value/MinimumMinimumflatten_7/Reshape:output:04tf.clip_by_value_10/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2+
)tf.clip_by_value_10/clip_by_value/Minimum?
#tf.clip_by_value_10/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2%
#tf.clip_by_value_10/clip_by_value/y?
!tf.clip_by_value_10/clip_by_valueMaximum-tf.clip_by_value_10/clip_by_value/Minimum:z:0,tf.clip_by_value_10/clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????2#
!tf.clip_by_value_10/clip_by_value?
#tf.compat.v1.floor_div_7/FloorDiv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2%
#tf.compat.v1.floor_div_7/FloorDiv/y?
!tf.compat.v1.floor_div_7/FloorDivFloorDiv%tf.clip_by_value_10/clip_by_value:z:0,tf.compat.v1.floor_div_7/FloorDiv/y:output:0*
T0*'
_output_shapes
:?????????2#
!tf.compat.v1.floor_div_7/FloorDiv?
%tf.math.greater_equal_10/GreaterEqualGreaterEqualflatten_7/Reshape:output:0'tf_math_greater_equal_10_greaterequal_y*
T0*'
_output_shapes
:?????????2'
%tf.math.greater_equal_10/GreaterEqual?
tf.math.floormod_7/FloorMod/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2
tf.math.floormod_7/FloorMod/y?
tf.math.floormod_7/FloorModFloorMod%tf.clip_by_value_10/clip_by_value:z:0&tf.math.floormod_7/FloorMod/y:output:0*
T0*'
_output_shapes
:?????????2
tf.math.floormod_7/FloorMod?
embedding_23/CastCast%tf.clip_by_value_10/clip_by_value:z:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2
embedding_23/Cast?
embedding_23/embedding_lookupResourceGather&embedding_23_embedding_lookup_20519328embedding_23/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*9
_class/
-+loc:@embedding_23/embedding_lookup/20519328*,
_output_shapes
:??????????*
dtype02
embedding_23/embedding_lookup?
&embedding_23/embedding_lookup/IdentityIdentity&embedding_23/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*9
_class/
-+loc:@embedding_23/embedding_lookup/20519328*,
_output_shapes
:??????????2(
&embedding_23/embedding_lookup/Identity?
(embedding_23/embedding_lookup/Identity_1Identity/embedding_23/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????2*
(embedding_23/embedding_lookup/Identity_1?
embedding_21/CastCast%tf.compat.v1.floor_div_7/FloorDiv:z:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2
embedding_21/Cast?
embedding_21/embedding_lookupResourceGather&embedding_21_embedding_lookup_20519334embedding_21/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*9
_class/
-+loc:@embedding_21/embedding_lookup/20519334*,
_output_shapes
:??????????*
dtype02
embedding_21/embedding_lookup?
&embedding_21/embedding_lookup/IdentityIdentity&embedding_21/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*9
_class/
-+loc:@embedding_21/embedding_lookup/20519334*,
_output_shapes
:??????????2(
&embedding_21/embedding_lookup/Identity?
(embedding_21/embedding_lookup/Identity_1Identity/embedding_21/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????2*
(embedding_21/embedding_lookup/Identity_1?
tf.cast_10/CastCast)tf.math.greater_equal_10/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2
tf.cast_10/Cast?
tf.__operators__.add_20/AddV2AddV21embedding_23/embedding_lookup/Identity_1:output:01embedding_21/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:??????????2
tf.__operators__.add_20/AddV2?
embedding_22/CastCasttf.math.floormod_7/FloorMod:z:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2
embedding_22/Cast?
embedding_22/embedding_lookupResourceGather&embedding_22_embedding_lookup_20519342embedding_22/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*9
_class/
-+loc:@embedding_22/embedding_lookup/20519342*,
_output_shapes
:??????????*
dtype02
embedding_22/embedding_lookup?
&embedding_22/embedding_lookup/IdentityIdentity&embedding_22/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*9
_class/
-+loc:@embedding_22/embedding_lookup/20519342*,
_output_shapes
:??????????2(
&embedding_22/embedding_lookup/Identity?
(embedding_22/embedding_lookup/Identity_1Identity/embedding_22/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????2*
(embedding_22/embedding_lookup/Identity_1?
tf.__operators__.add_21/AddV2AddV2!tf.__operators__.add_20/AddV2:z:01embedding_22/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:??????????2
tf.__operators__.add_21/AddV2?
tf.expand_dims_7/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2!
tf.expand_dims_7/ExpandDims/dim?
tf.expand_dims_7/ExpandDims
ExpandDimstf.cast_10/Cast:y:0(tf.expand_dims_7/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????2
tf.expand_dims_7/ExpandDims?
tf.math.multiply_7/MulMul!tf.__operators__.add_21/AddV2:z:0$tf.expand_dims_7/ExpandDims:output:0*
T0*,
_output_shapes
:??????????2
tf.math.multiply_7/Mul?
*tf.math.reduce_sum_7/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2,
*tf.math.reduce_sum_7/Sum/reduction_indices?
tf.math.reduce_sum_7/SumSumtf.math.multiply_7/Mul:z:03tf.math.reduce_sum_7/Sum/reduction_indices:output:0*
T0*(
_output_shapes
:??????????2
tf.math.reduce_sum_7/Sum?
IdentityIdentity!tf.math.reduce_sum_7/Sum:output:0^embedding_21/embedding_lookup^embedding_22/embedding_lookup^embedding_23/embedding_lookup*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:?????????: :::2>
embedding_21/embedding_lookupembedding_21/embedding_lookup2>
embedding_22/embedding_lookupembedding_22/embedding_lookup2>
embedding_23/embedding_lookupembedding_23/embedding_lookup:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: 
?	
?
F__inference_dense_27_layer_call_and_return_conditional_losses_20517921

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
?
?
+__inference_dense_29_layer_call_fn_20519459

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
F__inference_dense_29_layer_call_and_return_conditional_losses_205180012
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
F__inference_dense_35_layer_call_and_return_conditional_losses_20519571

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
?
?
1__inference_custom_model_3_layer_call_fn_20519160

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
L__inference_custom_model_3_layer_call_and_return_conditional_losses_205185382
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
?
c
G__inference_flatten_6_layer_call_and_return_conditional_losses_20519586

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
F__inference_dense_32_layer_call_and_return_conditional_losses_20518055

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
?
?
&__inference_signature_wrapper_20518682
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
#__inference__wrapped_model_205173712
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
?-
?
E__inference_model_7_layer_call_and_return_conditional_losses_20517700
input_8+
'tf_math_greater_equal_10_greaterequal_y
embedding_23_20517644
embedding_21_20517666
embedding_22_20517690
identity??$embedding_21/StatefulPartitionedCall?$embedding_22/StatefulPartitionedCall?$embedding_23/StatefulPartitionedCall?
flatten_7/PartitionedCallPartitionedCallinput_8*
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
GPU2 *0J 8? *P
fKRI
G__inference_flatten_7_layer_call_and_return_conditional_losses_205176072
flatten_7/PartitionedCall?
+tf.clip_by_value_10/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 * $tI2-
+tf.clip_by_value_10/clip_by_value/Minimum/y?
)tf.clip_by_value_10/clip_by_value/MinimumMinimum"flatten_7/PartitionedCall:output:04tf.clip_by_value_10/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2+
)tf.clip_by_value_10/clip_by_value/Minimum?
#tf.clip_by_value_10/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2%
#tf.clip_by_value_10/clip_by_value/y?
!tf.clip_by_value_10/clip_by_valueMaximum-tf.clip_by_value_10/clip_by_value/Minimum:z:0,tf.clip_by_value_10/clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????2#
!tf.clip_by_value_10/clip_by_value?
#tf.compat.v1.floor_div_7/FloorDiv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2%
#tf.compat.v1.floor_div_7/FloorDiv/y?
!tf.compat.v1.floor_div_7/FloorDivFloorDiv%tf.clip_by_value_10/clip_by_value:z:0,tf.compat.v1.floor_div_7/FloorDiv/y:output:0*
T0*'
_output_shapes
:?????????2#
!tf.compat.v1.floor_div_7/FloorDiv?
%tf.math.greater_equal_10/GreaterEqualGreaterEqual"flatten_7/PartitionedCall:output:0'tf_math_greater_equal_10_greaterequal_y*
T0*'
_output_shapes
:?????????2'
%tf.math.greater_equal_10/GreaterEqual?
tf.math.floormod_7/FloorMod/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2
tf.math.floormod_7/FloorMod/y?
tf.math.floormod_7/FloorModFloorMod%tf.clip_by_value_10/clip_by_value:z:0&tf.math.floormod_7/FloorMod/y:output:0*
T0*'
_output_shapes
:?????????2
tf.math.floormod_7/FloorMod?
$embedding_23/StatefulPartitionedCallStatefulPartitionedCall%tf.clip_by_value_10/clip_by_value:z:0embedding_23_20517644*
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
J__inference_embedding_23_layer_call_and_return_conditional_losses_205176352&
$embedding_23/StatefulPartitionedCall?
$embedding_21/StatefulPartitionedCallStatefulPartitionedCall%tf.compat.v1.floor_div_7/FloorDiv:z:0embedding_21_20517666*
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
J__inference_embedding_21_layer_call_and_return_conditional_losses_205176572&
$embedding_21/StatefulPartitionedCall?
tf.cast_10/CastCast)tf.math.greater_equal_10/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2
tf.cast_10/Cast?
tf.__operators__.add_20/AddV2AddV2-embedding_23/StatefulPartitionedCall:output:0-embedding_21/StatefulPartitionedCall:output:0*
T0*,
_output_shapes
:??????????2
tf.__operators__.add_20/AddV2?
$embedding_22/StatefulPartitionedCallStatefulPartitionedCalltf.math.floormod_7/FloorMod:z:0embedding_22_20517690*
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
J__inference_embedding_22_layer_call_and_return_conditional_losses_205176812&
$embedding_22/StatefulPartitionedCall?
tf.__operators__.add_21/AddV2AddV2!tf.__operators__.add_20/AddV2:z:0-embedding_22/StatefulPartitionedCall:output:0*
T0*,
_output_shapes
:??????????2
tf.__operators__.add_21/AddV2?
tf.expand_dims_7/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2!
tf.expand_dims_7/ExpandDims/dim?
tf.expand_dims_7/ExpandDims
ExpandDimstf.cast_10/Cast:y:0(tf.expand_dims_7/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????2
tf.expand_dims_7/ExpandDims?
tf.math.multiply_7/MulMul!tf.__operators__.add_21/AddV2:z:0$tf.expand_dims_7/ExpandDims:output:0*
T0*,
_output_shapes
:??????????2
tf.math.multiply_7/Mul?
*tf.math.reduce_sum_7/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2,
*tf.math.reduce_sum_7/Sum/reduction_indices?
tf.math.reduce_sum_7/SumSumtf.math.multiply_7/Mul:z:03tf.math.reduce_sum_7/Sum/reduction_indices:output:0*
T0*(
_output_shapes
:??????????2
tf.math.reduce_sum_7/Sum?
IdentityIdentity!tf.math.reduce_sum_7/Sum:output:0%^embedding_21/StatefulPartitionedCall%^embedding_22/StatefulPartitionedCall%^embedding_23/StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:?????????: :::2L
$embedding_21/StatefulPartitionedCall$embedding_21/StatefulPartitionedCall2L
$embedding_22/StatefulPartitionedCall$embedding_22/StatefulPartitionedCall2L
$embedding_23/StatefulPartitionedCall$embedding_23/StatefulPartitionedCall:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_8:

_output_shapes
: 
?
u
/__inference_embedding_19_layer_call_fn_20519642

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
J__inference_embedding_19_layer_call_and_return_conditional_losses_205174552
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
?
?
*__inference_model_6_layer_call_fn_20519270

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
GPU2 *0J 8? *N
fIRG
E__inference_model_6_layer_call_and_return_conditional_losses_205175862
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
?9
?
E__inference_model_7_layer_call_and_return_conditional_losses_20519312

inputs+
'tf_math_greater_equal_10_greaterequal_y*
&embedding_23_embedding_lookup_20519286*
&embedding_21_embedding_lookup_20519292*
&embedding_22_embedding_lookup_20519300
identity??embedding_21/embedding_lookup?embedding_22/embedding_lookup?embedding_23/embedding_lookups
flatten_7/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
flatten_7/Const?
flatten_7/ReshapeReshapeinputsflatten_7/Const:output:0*
T0*'
_output_shapes
:?????????2
flatten_7/Reshape?
+tf.clip_by_value_10/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 * $tI2-
+tf.clip_by_value_10/clip_by_value/Minimum/y?
)tf.clip_by_value_10/clip_by_value/MinimumMinimumflatten_7/Reshape:output:04tf.clip_by_value_10/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2+
)tf.clip_by_value_10/clip_by_value/Minimum?
#tf.clip_by_value_10/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2%
#tf.clip_by_value_10/clip_by_value/y?
!tf.clip_by_value_10/clip_by_valueMaximum-tf.clip_by_value_10/clip_by_value/Minimum:z:0,tf.clip_by_value_10/clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????2#
!tf.clip_by_value_10/clip_by_value?
#tf.compat.v1.floor_div_7/FloorDiv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2%
#tf.compat.v1.floor_div_7/FloorDiv/y?
!tf.compat.v1.floor_div_7/FloorDivFloorDiv%tf.clip_by_value_10/clip_by_value:z:0,tf.compat.v1.floor_div_7/FloorDiv/y:output:0*
T0*'
_output_shapes
:?????????2#
!tf.compat.v1.floor_div_7/FloorDiv?
%tf.math.greater_equal_10/GreaterEqualGreaterEqualflatten_7/Reshape:output:0'tf_math_greater_equal_10_greaterequal_y*
T0*'
_output_shapes
:?????????2'
%tf.math.greater_equal_10/GreaterEqual?
tf.math.floormod_7/FloorMod/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2
tf.math.floormod_7/FloorMod/y?
tf.math.floormod_7/FloorModFloorMod%tf.clip_by_value_10/clip_by_value:z:0&tf.math.floormod_7/FloorMod/y:output:0*
T0*'
_output_shapes
:?????????2
tf.math.floormod_7/FloorMod?
embedding_23/CastCast%tf.clip_by_value_10/clip_by_value:z:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2
embedding_23/Cast?
embedding_23/embedding_lookupResourceGather&embedding_23_embedding_lookup_20519286embedding_23/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*9
_class/
-+loc:@embedding_23/embedding_lookup/20519286*,
_output_shapes
:??????????*
dtype02
embedding_23/embedding_lookup?
&embedding_23/embedding_lookup/IdentityIdentity&embedding_23/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*9
_class/
-+loc:@embedding_23/embedding_lookup/20519286*,
_output_shapes
:??????????2(
&embedding_23/embedding_lookup/Identity?
(embedding_23/embedding_lookup/Identity_1Identity/embedding_23/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????2*
(embedding_23/embedding_lookup/Identity_1?
embedding_21/CastCast%tf.compat.v1.floor_div_7/FloorDiv:z:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2
embedding_21/Cast?
embedding_21/embedding_lookupResourceGather&embedding_21_embedding_lookup_20519292embedding_21/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*9
_class/
-+loc:@embedding_21/embedding_lookup/20519292*,
_output_shapes
:??????????*
dtype02
embedding_21/embedding_lookup?
&embedding_21/embedding_lookup/IdentityIdentity&embedding_21/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*9
_class/
-+loc:@embedding_21/embedding_lookup/20519292*,
_output_shapes
:??????????2(
&embedding_21/embedding_lookup/Identity?
(embedding_21/embedding_lookup/Identity_1Identity/embedding_21/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????2*
(embedding_21/embedding_lookup/Identity_1?
tf.cast_10/CastCast)tf.math.greater_equal_10/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2
tf.cast_10/Cast?
tf.__operators__.add_20/AddV2AddV21embedding_23/embedding_lookup/Identity_1:output:01embedding_21/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:??????????2
tf.__operators__.add_20/AddV2?
embedding_22/CastCasttf.math.floormod_7/FloorMod:z:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2
embedding_22/Cast?
embedding_22/embedding_lookupResourceGather&embedding_22_embedding_lookup_20519300embedding_22/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*9
_class/
-+loc:@embedding_22/embedding_lookup/20519300*,
_output_shapes
:??????????*
dtype02
embedding_22/embedding_lookup?
&embedding_22/embedding_lookup/IdentityIdentity&embedding_22/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*9
_class/
-+loc:@embedding_22/embedding_lookup/20519300*,
_output_shapes
:??????????2(
&embedding_22/embedding_lookup/Identity?
(embedding_22/embedding_lookup/Identity_1Identity/embedding_22/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????2*
(embedding_22/embedding_lookup/Identity_1?
tf.__operators__.add_21/AddV2AddV2!tf.__operators__.add_20/AddV2:z:01embedding_22/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:??????????2
tf.__operators__.add_21/AddV2?
tf.expand_dims_7/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2!
tf.expand_dims_7/ExpandDims/dim?
tf.expand_dims_7/ExpandDims
ExpandDimstf.cast_10/Cast:y:0(tf.expand_dims_7/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????2
tf.expand_dims_7/ExpandDims?
tf.math.multiply_7/MulMul!tf.__operators__.add_21/AddV2:z:0$tf.expand_dims_7/ExpandDims:output:0*
T0*,
_output_shapes
:??????????2
tf.math.multiply_7/Mul?
*tf.math.reduce_sum_7/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2,
*tf.math.reduce_sum_7/Sum/reduction_indices?
tf.math.reduce_sum_7/SumSumtf.math.multiply_7/Mul:z:03tf.math.reduce_sum_7/Sum/reduction_indices:output:0*
T0*(
_output_shapes
:??????????2
tf.math.reduce_sum_7/Sum?
IdentityIdentity!tf.math.reduce_sum_7/Sum:output:0^embedding_21/embedding_lookup^embedding_22/embedding_lookup^embedding_23/embedding_lookup*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:?????????: :::2>
embedding_21/embedding_lookupembedding_21/embedding_lookup2>
embedding_22/embedding_lookupembedding_22/embedding_lookup2>
embedding_23/embedding_lookupembedding_23/embedding_lookup:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: 
?
?
+__inference_dense_34_layer_call_fn_20519535

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
F__inference_dense_34_layer_call_and_return_conditional_losses_205181102
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
?
?
I__inference_normalize_3_layer_call_and_return_conditional_losses_20519552
x3
/normalization_3_reshape_readvariableop_resource5
1normalization_3_reshape_1_readvariableop_resource
identity??&normalization_3/Reshape/ReadVariableOp?(normalization_3/Reshape_1/ReadVariableOp?
&normalization_3/Reshape/ReadVariableOpReadVariableOp/normalization_3_reshape_readvariableop_resource*
_output_shapes	
:?*
dtype02(
&normalization_3/Reshape/ReadVariableOp?
normalization_3/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2
normalization_3/Reshape/shape?
normalization_3/ReshapeReshape.normalization_3/Reshape/ReadVariableOp:value:0&normalization_3/Reshape/shape:output:0*
T0*
_output_shapes
:	?2
normalization_3/Reshape?
(normalization_3/Reshape_1/ReadVariableOpReadVariableOp1normalization_3_reshape_1_readvariableop_resource*
_output_shapes	
:?*
dtype02*
(normalization_3/Reshape_1/ReadVariableOp?
normalization_3/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2!
normalization_3/Reshape_1/shape?
normalization_3/Reshape_1Reshape0normalization_3/Reshape_1/ReadVariableOp:value:0(normalization_3/Reshape_1/shape:output:0*
T0*
_output_shapes
:	?2
normalization_3/Reshape_1?
normalization_3/subSubx normalization_3/Reshape:output:0*
T0*(
_output_shapes
:??????????2
normalization_3/sub?
normalization_3/SqrtSqrt"normalization_3/Reshape_1:output:0*
T0*
_output_shapes
:	?2
normalization_3/Sqrt{
normalization_3/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32
normalization_3/Maximum/y?
normalization_3/MaximumMaximumnormalization_3/Sqrt:y:0"normalization_3/Maximum/y:output:0*
T0*
_output_shapes
:	?2
normalization_3/Maximum?
normalization_3/truedivRealDivnormalization_3/sub:z:0normalization_3/Maximum:z:0*
T0*(
_output_shapes
:??????????2
normalization_3/truediv?
IdentityIdentitynormalization_3/truediv:z:0'^normalization_3/Reshape/ReadVariableOp)^normalization_3/Reshape_1/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::2P
&normalization_3/Reshape/ReadVariableOp&normalization_3/Reshape/ReadVariableOp2T
(normalization_3/Reshape_1/ReadVariableOp(normalization_3/Reshape_1/ReadVariableOp:K G
(
_output_shapes
:??????????

_user_specified_namex
??
?#
!__inference__traced_save_20519980
file_prefix.
*savev2_dense_27_kernel_read_readvariableop,
(savev2_dense_27_bias_read_readvariableop.
*savev2_dense_28_kernel_read_readvariableop,
(savev2_dense_28_bias_read_readvariableop.
*savev2_dense_30_kernel_read_readvariableop,
(savev2_dense_30_bias_read_readvariableop.
*savev2_dense_29_kernel_read_readvariableop,
(savev2_dense_29_bias_read_readvariableop.
*savev2_dense_31_kernel_read_readvariableop,
(savev2_dense_31_bias_read_readvariableop.
*savev2_dense_32_kernel_read_readvariableop,
(savev2_dense_32_bias_read_readvariableop.
*savev2_dense_33_kernel_read_readvariableop,
(savev2_dense_33_bias_read_readvariableop.
*savev2_dense_34_kernel_read_readvariableop,
(savev2_dense_34_bias_read_readvariableop.
*savev2_dense_35_kernel_read_readvariableop,
(savev2_dense_35_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop6
2savev2_embedding_20_embeddings_read_readvariableop6
2savev2_embedding_18_embeddings_read_readvariableop6
2savev2_embedding_19_embeddings_read_readvariableop6
2savev2_embedding_23_embeddings_read_readvariableop6
2savev2_embedding_21_embeddings_read_readvariableop6
2savev2_embedding_22_embeddings_read_readvariableop?
;savev2_normalize_3_normalization_3_mean_read_readvariableopC
?savev2_normalize_3_normalization_3_variance_read_readvariableop@
<savev2_normalize_3_normalization_3_count_read_readvariableop	$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop5
1savev2_adam_dense_27_kernel_m_read_readvariableop3
/savev2_adam_dense_27_bias_m_read_readvariableop5
1savev2_adam_dense_28_kernel_m_read_readvariableop3
/savev2_adam_dense_28_bias_m_read_readvariableop5
1savev2_adam_dense_30_kernel_m_read_readvariableop3
/savev2_adam_dense_30_bias_m_read_readvariableop5
1savev2_adam_dense_29_kernel_m_read_readvariableop3
/savev2_adam_dense_29_bias_m_read_readvariableop5
1savev2_adam_dense_31_kernel_m_read_readvariableop3
/savev2_adam_dense_31_bias_m_read_readvariableop5
1savev2_adam_dense_32_kernel_m_read_readvariableop3
/savev2_adam_dense_32_bias_m_read_readvariableop5
1savev2_adam_dense_33_kernel_m_read_readvariableop3
/savev2_adam_dense_33_bias_m_read_readvariableop5
1savev2_adam_dense_34_kernel_m_read_readvariableop3
/savev2_adam_dense_34_bias_m_read_readvariableop5
1savev2_adam_dense_35_kernel_m_read_readvariableop3
/savev2_adam_dense_35_bias_m_read_readvariableop=
9savev2_adam_embedding_20_embeddings_m_read_readvariableop=
9savev2_adam_embedding_18_embeddings_m_read_readvariableop=
9savev2_adam_embedding_19_embeddings_m_read_readvariableop=
9savev2_adam_embedding_23_embeddings_m_read_readvariableop=
9savev2_adam_embedding_21_embeddings_m_read_readvariableop=
9savev2_adam_embedding_22_embeddings_m_read_readvariableop5
1savev2_adam_dense_27_kernel_v_read_readvariableop3
/savev2_adam_dense_27_bias_v_read_readvariableop5
1savev2_adam_dense_28_kernel_v_read_readvariableop3
/savev2_adam_dense_28_bias_v_read_readvariableop5
1savev2_adam_dense_30_kernel_v_read_readvariableop3
/savev2_adam_dense_30_bias_v_read_readvariableop5
1savev2_adam_dense_29_kernel_v_read_readvariableop3
/savev2_adam_dense_29_bias_v_read_readvariableop5
1savev2_adam_dense_31_kernel_v_read_readvariableop3
/savev2_adam_dense_31_bias_v_read_readvariableop5
1savev2_adam_dense_32_kernel_v_read_readvariableop3
/savev2_adam_dense_32_bias_v_read_readvariableop5
1savev2_adam_dense_33_kernel_v_read_readvariableop3
/savev2_adam_dense_33_bias_v_read_readvariableop5
1savev2_adam_dense_34_kernel_v_read_readvariableop3
/savev2_adam_dense_34_bias_v_read_readvariableop5
1savev2_adam_dense_35_kernel_v_read_readvariableop3
/savev2_adam_dense_35_bias_v_read_readvariableop=
9savev2_adam_embedding_20_embeddings_v_read_readvariableop=
9savev2_adam_embedding_18_embeddings_v_read_readvariableop=
9savev2_adam_embedding_19_embeddings_v_read_readvariableop=
9savev2_adam_embedding_23_embeddings_v_read_readvariableop=
9savev2_adam_embedding_21_embeddings_v_read_readvariableop=
9savev2_adam_embedding_22_embeddings_v_read_readvariableop
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
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_dense_27_kernel_read_readvariableop(savev2_dense_27_bias_read_readvariableop*savev2_dense_28_kernel_read_readvariableop(savev2_dense_28_bias_read_readvariableop*savev2_dense_30_kernel_read_readvariableop(savev2_dense_30_bias_read_readvariableop*savev2_dense_29_kernel_read_readvariableop(savev2_dense_29_bias_read_readvariableop*savev2_dense_31_kernel_read_readvariableop(savev2_dense_31_bias_read_readvariableop*savev2_dense_32_kernel_read_readvariableop(savev2_dense_32_bias_read_readvariableop*savev2_dense_33_kernel_read_readvariableop(savev2_dense_33_bias_read_readvariableop*savev2_dense_34_kernel_read_readvariableop(savev2_dense_34_bias_read_readvariableop*savev2_dense_35_kernel_read_readvariableop(savev2_dense_35_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop2savev2_embedding_20_embeddings_read_readvariableop2savev2_embedding_18_embeddings_read_readvariableop2savev2_embedding_19_embeddings_read_readvariableop2savev2_embedding_23_embeddings_read_readvariableop2savev2_embedding_21_embeddings_read_readvariableop2savev2_embedding_22_embeddings_read_readvariableop;savev2_normalize_3_normalization_3_mean_read_readvariableop?savev2_normalize_3_normalization_3_variance_read_readvariableop<savev2_normalize_3_normalization_3_count_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop1savev2_adam_dense_27_kernel_m_read_readvariableop/savev2_adam_dense_27_bias_m_read_readvariableop1savev2_adam_dense_28_kernel_m_read_readvariableop/savev2_adam_dense_28_bias_m_read_readvariableop1savev2_adam_dense_30_kernel_m_read_readvariableop/savev2_adam_dense_30_bias_m_read_readvariableop1savev2_adam_dense_29_kernel_m_read_readvariableop/savev2_adam_dense_29_bias_m_read_readvariableop1savev2_adam_dense_31_kernel_m_read_readvariableop/savev2_adam_dense_31_bias_m_read_readvariableop1savev2_adam_dense_32_kernel_m_read_readvariableop/savev2_adam_dense_32_bias_m_read_readvariableop1savev2_adam_dense_33_kernel_m_read_readvariableop/savev2_adam_dense_33_bias_m_read_readvariableop1savev2_adam_dense_34_kernel_m_read_readvariableop/savev2_adam_dense_34_bias_m_read_readvariableop1savev2_adam_dense_35_kernel_m_read_readvariableop/savev2_adam_dense_35_bias_m_read_readvariableop9savev2_adam_embedding_20_embeddings_m_read_readvariableop9savev2_adam_embedding_18_embeddings_m_read_readvariableop9savev2_adam_embedding_19_embeddings_m_read_readvariableop9savev2_adam_embedding_23_embeddings_m_read_readvariableop9savev2_adam_embedding_21_embeddings_m_read_readvariableop9savev2_adam_embedding_22_embeddings_m_read_readvariableop1savev2_adam_dense_27_kernel_v_read_readvariableop/savev2_adam_dense_27_bias_v_read_readvariableop1savev2_adam_dense_28_kernel_v_read_readvariableop/savev2_adam_dense_28_bias_v_read_readvariableop1savev2_adam_dense_30_kernel_v_read_readvariableop/savev2_adam_dense_30_bias_v_read_readvariableop1savev2_adam_dense_29_kernel_v_read_readvariableop/savev2_adam_dense_29_bias_v_read_readvariableop1savev2_adam_dense_31_kernel_v_read_readvariableop/savev2_adam_dense_31_bias_v_read_readvariableop1savev2_adam_dense_32_kernel_v_read_readvariableop/savev2_adam_dense_32_bias_v_read_readvariableop1savev2_adam_dense_33_kernel_v_read_readvariableop/savev2_adam_dense_33_bias_v_read_readvariableop1savev2_adam_dense_34_kernel_v_read_readvariableop/savev2_adam_dense_34_bias_v_read_readvariableop1savev2_adam_dense_35_kernel_v_read_readvariableop/savev2_adam_dense_35_bias_v_read_readvariableop9savev2_adam_embedding_20_embeddings_v_read_readvariableop9savev2_adam_embedding_18_embeddings_v_read_readvariableop9savev2_adam_embedding_19_embeddings_v_read_readvariableop9savev2_adam_embedding_23_embeddings_v_read_readvariableop9savev2_adam_embedding_21_embeddings_v_read_readvariableop9savev2_adam_embedding_22_embeddings_v_read_readvariableopsavev2_const_5"/device:CPU:0*
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
: "?L
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
dense_350
StatefulPartitionedCall:0?????????tensorflow/serving/predict:??
Ϣ
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
regularization_losses
	variables
 	keras_api
!
signatures
?_default_save_signature
+?&call_and_return_all_conditional_losses
?__call__"ڛ
_tf_keras_network??{"class_name": "CustomModel", "name": "custom_model_3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "custom_model_3", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 2]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "cards0"}, "name": "cards0", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "cards1"}, "name": "cards1", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 10]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "bets"}, "name": "bets", "inbound_nodes": []}, {"class_name": "Functional", "config": {"name": "model_6", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 2]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_7"}, "name": "input_7", "inbound_nodes": []}, {"class_name": "Flatten", "config": {"name": "flatten_6", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_6", "inbound_nodes": [[["input_7", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.clip_by_value_9", "trainable": true, "dtype": "float32", "function": "clip_by_value"}, "name": "tf.clip_by_value_9", "inbound_nodes": [["flatten_6", 0, 0, {"clip_value_min": 0, "clip_value_max": 1000000.0}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.compat.v1.floor_div_6", "trainable": true, "dtype": "float32", "function": "compat.v1.floor_div"}, "name": "tf.compat.v1.floor_div_6", "inbound_nodes": [["tf.clip_by_value_9", 0, 0, {"y": 4, "name": null}]]}, {"class_name": "Embedding", "config": {"name": "embedding_20", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 52, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding_20", "inbound_nodes": [[["tf.clip_by_value_9", 0, 0, {}]]]}, {"class_name": "Embedding", "config": {"name": "embedding_18", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 13, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding_18", "inbound_nodes": [[["tf.compat.v1.floor_div_6", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.floormod_6", "trainable": true, "dtype": "float32", "function": "math.floormod"}, "name": "tf.math.floormod_6", "inbound_nodes": [["tf.clip_by_value_9", 0, 0, {"y": 4, "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.greater_equal_9", "trainable": true, "dtype": "float32", "function": "math.greater_equal"}, "name": "tf.math.greater_equal_9", "inbound_nodes": [["flatten_6", 0, 0, {"y": 0.0, "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_18", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_18", "inbound_nodes": [["embedding_20", 0, 0, {"y": ["embedding_18", 0, 0], "name": null}]]}, {"class_name": "Embedding", "config": {"name": "embedding_19", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 4, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding_19", "inbound_nodes": [[["tf.math.floormod_6", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.cast_9", "trainable": true, "dtype": "float32", "function": "cast"}, "name": "tf.cast_9", "inbound_nodes": [["tf.math.greater_equal_9", 0, 0, {"dtype": "float32"}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_19", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_19", "inbound_nodes": [["tf.__operators__.add_18", 0, 0, {"y": ["embedding_19", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.expand_dims_6", "trainable": true, "dtype": "float32", "function": "expand_dims"}, "name": "tf.expand_dims_6", "inbound_nodes": [["tf.cast_9", 0, 0, {"axis": -1}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.multiply_6", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "name": "tf.math.multiply_6", "inbound_nodes": [["tf.__operators__.add_19", 0, 0, {"y": ["tf.expand_dims_6", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.reduce_sum_6", "trainable": true, "dtype": "float32", "function": "math.reduce_sum"}, "name": "tf.math.reduce_sum_6", "inbound_nodes": [["tf.math.multiply_6", 0, 0, {"axis": 1}]]}], "input_layers": [["input_7", 0, 0]], "output_layers": [["tf.math.reduce_sum_6", 0, 0]]}, "name": "model_6", "inbound_nodes": [[["cards0", 0, 0, {}]]]}, {"class_name": "Functional", "config": {"name": "model_7", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_8"}, "name": "input_8", "inbound_nodes": []}, {"class_name": "Flatten", "config": {"name": "flatten_7", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_7", "inbound_nodes": [[["input_8", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.clip_by_value_10", "trainable": true, "dtype": "float32", "function": "clip_by_value"}, "name": "tf.clip_by_value_10", "inbound_nodes": [["flatten_7", 0, 0, {"clip_value_min": 0, "clip_value_max": 1000000.0}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.compat.v1.floor_div_7", "trainable": true, "dtype": "float32", "function": "compat.v1.floor_div"}, "name": "tf.compat.v1.floor_div_7", "inbound_nodes": [["tf.clip_by_value_10", 0, 0, {"y": 4, "name": null}]]}, {"class_name": "Embedding", "config": {"name": "embedding_23", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 52, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding_23", "inbound_nodes": [[["tf.clip_by_value_10", 0, 0, {}]]]}, {"class_name": "Embedding", "config": {"name": "embedding_21", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 13, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding_21", "inbound_nodes": [[["tf.compat.v1.floor_div_7", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.floormod_7", "trainable": true, "dtype": "float32", "function": "math.floormod"}, "name": "tf.math.floormod_7", "inbound_nodes": [["tf.clip_by_value_10", 0, 0, {"y": 4, "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.greater_equal_10", "trainable": true, "dtype": "float32", "function": "math.greater_equal"}, "name": "tf.math.greater_equal_10", "inbound_nodes": [["flatten_7", 0, 0, {"y": 0.0, "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_20", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_20", "inbound_nodes": [["embedding_23", 0, 0, {"y": ["embedding_21", 0, 0], "name": null}]]}, {"class_name": "Embedding", "config": {"name": "embedding_22", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 4, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding_22", "inbound_nodes": [[["tf.math.floormod_7", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.cast_10", "trainable": true, "dtype": "float32", "function": "cast"}, "name": "tf.cast_10", "inbound_nodes": [["tf.math.greater_equal_10", 0, 0, {"dtype": "float32"}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_21", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_21", "inbound_nodes": [["tf.__operators__.add_20", 0, 0, {"y": ["embedding_22", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.expand_dims_7", "trainable": true, "dtype": "float32", "function": "expand_dims"}, "name": "tf.expand_dims_7", "inbound_nodes": [["tf.cast_10", 0, 0, {"axis": -1}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.multiply_7", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "name": "tf.math.multiply_7", "inbound_nodes": [["tf.__operators__.add_21", 0, 0, {"y": ["tf.expand_dims_7", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.reduce_sum_7", "trainable": true, "dtype": "float32", "function": "math.reduce_sum"}, "name": "tf.math.reduce_sum_7", "inbound_nodes": [["tf.math.multiply_7", 0, 0, {"axis": 1}]]}], "input_layers": [["input_8", 0, 0]], "output_layers": [["tf.math.reduce_sum_7", 0, 0]]}, "name": "model_7", "inbound_nodes": [[["cards1", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.greater_equal_11", "trainable": true, "dtype": "float32", "function": "math.greater_equal"}, "name": "tf.math.greater_equal_11", "inbound_nodes": [["bets", 0, 0, {"y": 0.0, "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.concat_9", "trainable": true, "dtype": "float32", "function": "concat"}, "name": "tf.concat_9", "inbound_nodes": [[["model_6", 1, 0, {"axis": 1}], ["model_7", 1, 0, {"axis": 1}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.clip_by_value_11", "trainable": true, "dtype": "float32", "function": "clip_by_value"}, "name": "tf.clip_by_value_11", "inbound_nodes": [["bets", 0, 0, {"clip_value_min": 0.0, "clip_value_max": 1000000.0}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.cast_11", "trainable": true, "dtype": "float32", "function": "cast"}, "name": "tf.cast_11", "inbound_nodes": [["tf.math.greater_equal_11", 0, 0, {"dtype": "float32"}]]}, {"class_name": "Dense", "config": {"name": "dense_27", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_27", "inbound_nodes": [[["tf.concat_9", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.concat_10", "trainable": true, "dtype": "float32", "function": "concat"}, "name": "tf.concat_10", "inbound_nodes": [[["tf.clip_by_value_11", 0, 0, {"axis": -1}], ["tf.cast_11", 0, 0, {"axis": -1}]]]}, {"class_name": "Dense", "config": {"name": "dense_28", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_28", "inbound_nodes": [[["dense_27", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_30", "trainable": true, "dtype": "float32", "units": 256, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_30", "inbound_nodes": [[["tf.concat_10", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_29", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_29", "inbound_nodes": [[["dense_28", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_31", "trainable": true, "dtype": "float32", "units": 256, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_31", "inbound_nodes": [[["dense_30", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.concat_11", "trainable": true, "dtype": "float32", "function": "concat"}, "name": "tf.concat_11", "inbound_nodes": [[["dense_29", 0, 0, {"axis": -1}], ["dense_31", 0, 0, {"axis": -1}]]]}, {"class_name": "Dense", "config": {"name": "dense_32", "trainable": true, "dtype": "float32", "units": 256, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_32", "inbound_nodes": [[["tf.concat_11", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.nn.relu_9", "trainable": true, "dtype": "float32", "function": "nn.relu"}, "name": "tf.nn.relu_9", "inbound_nodes": [["dense_32", 0, 0, {"name": null}]]}, {"class_name": "Dense", "config": {"name": "dense_33", "trainable": true, "dtype": "float32", "units": 256, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_33", "inbound_nodes": [[["tf.nn.relu_9", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_22", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_22", "inbound_nodes": [["dense_33", 0, 0, {"y": ["tf.nn.relu_9", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.nn.relu_10", "trainable": true, "dtype": "float32", "function": "nn.relu"}, "name": "tf.nn.relu_10", "inbound_nodes": [["tf.__operators__.add_22", 0, 0, {"name": null}]]}, {"class_name": "Dense", "config": {"name": "dense_34", "trainable": true, "dtype": "float32", "units": 256, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_34", "inbound_nodes": [[["tf.nn.relu_10", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_23", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_23", "inbound_nodes": [["dense_34", 0, 0, {"y": ["tf.nn.relu_10", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.nn.relu_11", "trainable": true, "dtype": "float32", "function": "nn.relu"}, "name": "tf.nn.relu_11", "inbound_nodes": [["tf.__operators__.add_23", 0, 0, {"name": null}]]}, {"class_name": "Normalize", "config": {"name": "normalize_3", "trainable": true, "dtype": "float32"}, "name": "normalize_3", "inbound_nodes": [[["tf.nn.relu_11", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_35", "trainable": true, "dtype": "float32", "units": 4, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Constant", "config": {"value": 0}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_35", "inbound_nodes": [[["normalize_3", 0, 0, {}]]]}], "input_layers": [[["cards0", 0, 0], ["cards1", 0, 0]], ["bets", 0, 0]], "output_layers": [["dense_35", 0, 0]]}, "build_input_shape": [[{"class_name": "TensorShape", "items": [null, 2]}, {"class_name": "TensorShape", "items": [null, 3]}], {"class_name": "TensorShape", "items": [null, 10]}], "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "CustomModel", "config": {"name": "custom_model_3", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 2]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "cards0"}, "name": "cards0", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "cards1"}, "name": "cards1", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 10]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "bets"}, "name": "bets", "inbound_nodes": []}, {"class_name": "Functional", "config": {"name": "model_6", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 2]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_7"}, "name": "input_7", "inbound_nodes": []}, {"class_name": "Flatten", "config": {"name": "flatten_6", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_6", "inbound_nodes": [[["input_7", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.clip_by_value_9", "trainable": true, "dtype": "float32", "function": "clip_by_value"}, "name": "tf.clip_by_value_9", "inbound_nodes": [["flatten_6", 0, 0, {"clip_value_min": 0, "clip_value_max": 1000000.0}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.compat.v1.floor_div_6", "trainable": true, "dtype": "float32", "function": "compat.v1.floor_div"}, "name": "tf.compat.v1.floor_div_6", "inbound_nodes": [["tf.clip_by_value_9", 0, 0, {"y": 4, "name": null}]]}, {"class_name": "Embedding", "config": {"name": "embedding_20", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 52, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding_20", "inbound_nodes": [[["tf.clip_by_value_9", 0, 0, {}]]]}, {"class_name": "Embedding", "config": {"name": "embedding_18", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 13, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding_18", "inbound_nodes": [[["tf.compat.v1.floor_div_6", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.floormod_6", "trainable": true, "dtype": "float32", "function": "math.floormod"}, "name": "tf.math.floormod_6", "inbound_nodes": [["tf.clip_by_value_9", 0, 0, {"y": 4, "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.greater_equal_9", "trainable": true, "dtype": "float32", "function": "math.greater_equal"}, "name": "tf.math.greater_equal_9", "inbound_nodes": [["flatten_6", 0, 0, {"y": 0.0, "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_18", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_18", "inbound_nodes": [["embedding_20", 0, 0, {"y": ["embedding_18", 0, 0], "name": null}]]}, {"class_name": "Embedding", "config": {"name": "embedding_19", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 4, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding_19", "inbound_nodes": [[["tf.math.floormod_6", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.cast_9", "trainable": true, "dtype": "float32", "function": "cast"}, "name": "tf.cast_9", "inbound_nodes": [["tf.math.greater_equal_9", 0, 0, {"dtype": "float32"}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_19", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_19", "inbound_nodes": [["tf.__operators__.add_18", 0, 0, {"y": ["embedding_19", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.expand_dims_6", "trainable": true, "dtype": "float32", "function": "expand_dims"}, "name": "tf.expand_dims_6", "inbound_nodes": [["tf.cast_9", 0, 0, {"axis": -1}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.multiply_6", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "name": "tf.math.multiply_6", "inbound_nodes": [["tf.__operators__.add_19", 0, 0, {"y": ["tf.expand_dims_6", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.reduce_sum_6", "trainable": true, "dtype": "float32", "function": "math.reduce_sum"}, "name": "tf.math.reduce_sum_6", "inbound_nodes": [["tf.math.multiply_6", 0, 0, {"axis": 1}]]}], "input_layers": [["input_7", 0, 0]], "output_layers": [["tf.math.reduce_sum_6", 0, 0]]}, "name": "model_6", "inbound_nodes": [[["cards0", 0, 0, {}]]]}, {"class_name": "Functional", "config": {"name": "model_7", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_8"}, "name": "input_8", "inbound_nodes": []}, {"class_name": "Flatten", "config": {"name": "flatten_7", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_7", "inbound_nodes": [[["input_8", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.clip_by_value_10", "trainable": true, "dtype": "float32", "function": "clip_by_value"}, "name": "tf.clip_by_value_10", "inbound_nodes": [["flatten_7", 0, 0, {"clip_value_min": 0, "clip_value_max": 1000000.0}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.compat.v1.floor_div_7", "trainable": true, "dtype": "float32", "function": "compat.v1.floor_div"}, "name": "tf.compat.v1.floor_div_7", "inbound_nodes": [["tf.clip_by_value_10", 0, 0, {"y": 4, "name": null}]]}, {"class_name": "Embedding", "config": {"name": "embedding_23", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 52, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding_23", "inbound_nodes": [[["tf.clip_by_value_10", 0, 0, {}]]]}, {"class_name": "Embedding", "config": {"name": "embedding_21", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 13, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding_21", "inbound_nodes": [[["tf.compat.v1.floor_div_7", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.floormod_7", "trainable": true, "dtype": "float32", "function": "math.floormod"}, "name": "tf.math.floormod_7", "inbound_nodes": [["tf.clip_by_value_10", 0, 0, {"y": 4, "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.greater_equal_10", "trainable": true, "dtype": "float32", "function": "math.greater_equal"}, "name": "tf.math.greater_equal_10", "inbound_nodes": [["flatten_7", 0, 0, {"y": 0.0, "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_20", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_20", "inbound_nodes": [["embedding_23", 0, 0, {"y": ["embedding_21", 0, 0], "name": null}]]}, {"class_name": "Embedding", "config": {"name": "embedding_22", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 4, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding_22", "inbound_nodes": [[["tf.math.floormod_7", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.cast_10", "trainable": true, "dtype": "float32", "function": "cast"}, "name": "tf.cast_10", "inbound_nodes": [["tf.math.greater_equal_10", 0, 0, {"dtype": "float32"}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_21", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_21", "inbound_nodes": [["tf.__operators__.add_20", 0, 0, {"y": ["embedding_22", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.expand_dims_7", "trainable": true, "dtype": "float32", "function": "expand_dims"}, "name": "tf.expand_dims_7", "inbound_nodes": [["tf.cast_10", 0, 0, {"axis": -1}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.multiply_7", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "name": "tf.math.multiply_7", "inbound_nodes": [["tf.__operators__.add_21", 0, 0, {"y": ["tf.expand_dims_7", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.reduce_sum_7", "trainable": true, "dtype": "float32", "function": "math.reduce_sum"}, "name": "tf.math.reduce_sum_7", "inbound_nodes": [["tf.math.multiply_7", 0, 0, {"axis": 1}]]}], "input_layers": [["input_8", 0, 0]], "output_layers": [["tf.math.reduce_sum_7", 0, 0]]}, "name": "model_7", "inbound_nodes": [[["cards1", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.greater_equal_11", "trainable": true, "dtype": "float32", "function": "math.greater_equal"}, "name": "tf.math.greater_equal_11", "inbound_nodes": [["bets", 0, 0, {"y": 0.0, "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.concat_9", "trainable": true, "dtype": "float32", "function": "concat"}, "name": "tf.concat_9", "inbound_nodes": [[["model_6", 1, 0, {"axis": 1}], ["model_7", 1, 0, {"axis": 1}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.clip_by_value_11", "trainable": true, "dtype": "float32", "function": "clip_by_value"}, "name": "tf.clip_by_value_11", "inbound_nodes": [["bets", 0, 0, {"clip_value_min": 0.0, "clip_value_max": 1000000.0}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.cast_11", "trainable": true, "dtype": "float32", "function": "cast"}, "name": "tf.cast_11", "inbound_nodes": [["tf.math.greater_equal_11", 0, 0, {"dtype": "float32"}]]}, {"class_name": "Dense", "config": {"name": "dense_27", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_27", "inbound_nodes": [[["tf.concat_9", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.concat_10", "trainable": true, "dtype": "float32", "function": "concat"}, "name": "tf.concat_10", "inbound_nodes": [[["tf.clip_by_value_11", 0, 0, {"axis": -1}], ["tf.cast_11", 0, 0, {"axis": -1}]]]}, {"class_name": "Dense", "config": {"name": "dense_28", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_28", "inbound_nodes": [[["dense_27", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_30", "trainable": true, "dtype": "float32", "units": 256, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_30", "inbound_nodes": [[["tf.concat_10", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_29", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_29", "inbound_nodes": [[["dense_28", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_31", "trainable": true, "dtype": "float32", "units": 256, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_31", "inbound_nodes": [[["dense_30", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.concat_11", "trainable": true, "dtype": "float32", "function": "concat"}, "name": "tf.concat_11", "inbound_nodes": [[["dense_29", 0, 0, {"axis": -1}], ["dense_31", 0, 0, {"axis": -1}]]]}, {"class_name": "Dense", "config": {"name": "dense_32", "trainable": true, "dtype": "float32", "units": 256, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_32", "inbound_nodes": [[["tf.concat_11", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.nn.relu_9", "trainable": true, "dtype": "float32", "function": "nn.relu"}, "name": "tf.nn.relu_9", "inbound_nodes": [["dense_32", 0, 0, {"name": null}]]}, {"class_name": "Dense", "config": {"name": "dense_33", "trainable": true, "dtype": "float32", "units": 256, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_33", "inbound_nodes": [[["tf.nn.relu_9", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_22", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_22", "inbound_nodes": [["dense_33", 0, 0, {"y": ["tf.nn.relu_9", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.nn.relu_10", "trainable": true, "dtype": "float32", "function": "nn.relu"}, "name": "tf.nn.relu_10", "inbound_nodes": [["tf.__operators__.add_22", 0, 0, {"name": null}]]}, {"class_name": "Dense", "config": {"name": "dense_34", "trainable": true, "dtype": "float32", "units": 256, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_34", "inbound_nodes": [[["tf.nn.relu_10", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_23", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_23", "inbound_nodes": [["dense_34", 0, 0, {"y": ["tf.nn.relu_10", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.nn.relu_11", "trainable": true, "dtype": "float32", "function": "nn.relu"}, "name": "tf.nn.relu_11", "inbound_nodes": [["tf.__operators__.add_23", 0, 0, {"name": null}]]}, {"class_name": "Normalize", "config": {"name": "normalize_3", "trainable": true, "dtype": "float32"}, "name": "normalize_3", "inbound_nodes": [[["tf.nn.relu_11", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_35", "trainable": true, "dtype": "float32", "units": 4, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Constant", "config": {"value": 0}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_35", "inbound_nodes": [[["normalize_3", 0, 0, {}]]]}], "input_layers": [[["cards0", 0, 0], ["cards1", 0, 0]], ["bets", 0, 0]], "output_layers": [["dense_35", 0, 0]]}}, "training_config": {"loss": null, "metrics": null, "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0020000000949949026, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "cards0", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 2]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 2]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "cards0"}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "cards1", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 3]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "cards1"}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "bets", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 10]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 10]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "bets"}}
?Q
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
2regularization_losses
3	variables
4	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?N
_tf_keras_network?N{"class_name": "Functional", "name": "model_6", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "model_6", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 2]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_7"}, "name": "input_7", "inbound_nodes": []}, {"class_name": "Flatten", "config": {"name": "flatten_6", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_6", "inbound_nodes": [[["input_7", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.clip_by_value_9", "trainable": true, "dtype": "float32", "function": "clip_by_value"}, "name": "tf.clip_by_value_9", "inbound_nodes": [["flatten_6", 0, 0, {"clip_value_min": 0, "clip_value_max": 1000000.0}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.compat.v1.floor_div_6", "trainable": true, "dtype": "float32", "function": "compat.v1.floor_div"}, "name": "tf.compat.v1.floor_div_6", "inbound_nodes": [["tf.clip_by_value_9", 0, 0, {"y": 4, "name": null}]]}, {"class_name": "Embedding", "config": {"name": "embedding_20", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 52, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding_20", "inbound_nodes": [[["tf.clip_by_value_9", 0, 0, {}]]]}, {"class_name": "Embedding", "config": {"name": "embedding_18", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 13, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding_18", "inbound_nodes": [[["tf.compat.v1.floor_div_6", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.floormod_6", "trainable": true, "dtype": "float32", "function": "math.floormod"}, "name": "tf.math.floormod_6", "inbound_nodes": [["tf.clip_by_value_9", 0, 0, {"y": 4, "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.greater_equal_9", "trainable": true, "dtype": "float32", "function": "math.greater_equal"}, "name": "tf.math.greater_equal_9", "inbound_nodes": [["flatten_6", 0, 0, {"y": 0.0, "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_18", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_18", "inbound_nodes": [["embedding_20", 0, 0, {"y": ["embedding_18", 0, 0], "name": null}]]}, {"class_name": "Embedding", "config": {"name": "embedding_19", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 4, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding_19", "inbound_nodes": [[["tf.math.floormod_6", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.cast_9", "trainable": true, "dtype": "float32", "function": "cast"}, "name": "tf.cast_9", "inbound_nodes": [["tf.math.greater_equal_9", 0, 0, {"dtype": "float32"}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_19", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_19", "inbound_nodes": [["tf.__operators__.add_18", 0, 0, {"y": ["embedding_19", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.expand_dims_6", "trainable": true, "dtype": "float32", "function": "expand_dims"}, "name": "tf.expand_dims_6", "inbound_nodes": [["tf.cast_9", 0, 0, {"axis": -1}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.multiply_6", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "name": "tf.math.multiply_6", "inbound_nodes": [["tf.__operators__.add_19", 0, 0, {"y": ["tf.expand_dims_6", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.reduce_sum_6", "trainable": true, "dtype": "float32", "function": "math.reduce_sum"}, "name": "tf.math.reduce_sum_6", "inbound_nodes": [["tf.math.multiply_6", 0, 0, {"axis": 1}]]}], "input_layers": [["input_7", 0, 0]], "output_layers": [["tf.math.reduce_sum_6", 0, 0]]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 2]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 2]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "model_6", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 2]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_7"}, "name": "input_7", "inbound_nodes": []}, {"class_name": "Flatten", "config": {"name": "flatten_6", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_6", "inbound_nodes": [[["input_7", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.clip_by_value_9", "trainable": true, "dtype": "float32", "function": "clip_by_value"}, "name": "tf.clip_by_value_9", "inbound_nodes": [["flatten_6", 0, 0, {"clip_value_min": 0, "clip_value_max": 1000000.0}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.compat.v1.floor_div_6", "trainable": true, "dtype": "float32", "function": "compat.v1.floor_div"}, "name": "tf.compat.v1.floor_div_6", "inbound_nodes": [["tf.clip_by_value_9", 0, 0, {"y": 4, "name": null}]]}, {"class_name": "Embedding", "config": {"name": "embedding_20", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 52, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding_20", "inbound_nodes": [[["tf.clip_by_value_9", 0, 0, {}]]]}, {"class_name": "Embedding", "config": {"name": "embedding_18", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 13, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding_18", "inbound_nodes": [[["tf.compat.v1.floor_div_6", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.floormod_6", "trainable": true, "dtype": "float32", "function": "math.floormod"}, "name": "tf.math.floormod_6", "inbound_nodes": [["tf.clip_by_value_9", 0, 0, {"y": 4, "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.greater_equal_9", "trainable": true, "dtype": "float32", "function": "math.greater_equal"}, "name": "tf.math.greater_equal_9", "inbound_nodes": [["flatten_6", 0, 0, {"y": 0.0, "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_18", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_18", "inbound_nodes": [["embedding_20", 0, 0, {"y": ["embedding_18", 0, 0], "name": null}]]}, {"class_name": "Embedding", "config": {"name": "embedding_19", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 4, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding_19", "inbound_nodes": [[["tf.math.floormod_6", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.cast_9", "trainable": true, "dtype": "float32", "function": "cast"}, "name": "tf.cast_9", "inbound_nodes": [["tf.math.greater_equal_9", 0, 0, {"dtype": "float32"}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_19", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_19", "inbound_nodes": [["tf.__operators__.add_18", 0, 0, {"y": ["embedding_19", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.expand_dims_6", "trainable": true, "dtype": "float32", "function": "expand_dims"}, "name": "tf.expand_dims_6", "inbound_nodes": [["tf.cast_9", 0, 0, {"axis": -1}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.multiply_6", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "name": "tf.math.multiply_6", "inbound_nodes": [["tf.__operators__.add_19", 0, 0, {"y": ["tf.expand_dims_6", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.reduce_sum_6", "trainable": true, "dtype": "float32", "function": "math.reduce_sum"}, "name": "tf.math.reduce_sum_6", "inbound_nodes": [["tf.math.multiply_6", 0, 0, {"axis": 1}]]}], "input_layers": [["input_7", 0, 0]], "output_layers": [["tf.math.reduce_sum_6", 0, 0]]}}}
?Q
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
Eregularization_losses
F	variables
G	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?N
_tf_keras_network?N{"class_name": "Functional", "name": "model_7", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "model_7", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_8"}, "name": "input_8", "inbound_nodes": []}, {"class_name": "Flatten", "config": {"name": "flatten_7", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_7", "inbound_nodes": [[["input_8", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.clip_by_value_10", "trainable": true, "dtype": "float32", "function": "clip_by_value"}, "name": "tf.clip_by_value_10", "inbound_nodes": [["flatten_7", 0, 0, {"clip_value_min": 0, "clip_value_max": 1000000.0}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.compat.v1.floor_div_7", "trainable": true, "dtype": "float32", "function": "compat.v1.floor_div"}, "name": "tf.compat.v1.floor_div_7", "inbound_nodes": [["tf.clip_by_value_10", 0, 0, {"y": 4, "name": null}]]}, {"class_name": "Embedding", "config": {"name": "embedding_23", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 52, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding_23", "inbound_nodes": [[["tf.clip_by_value_10", 0, 0, {}]]]}, {"class_name": "Embedding", "config": {"name": "embedding_21", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 13, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding_21", "inbound_nodes": [[["tf.compat.v1.floor_div_7", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.floormod_7", "trainable": true, "dtype": "float32", "function": "math.floormod"}, "name": "tf.math.floormod_7", "inbound_nodes": [["tf.clip_by_value_10", 0, 0, {"y": 4, "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.greater_equal_10", "trainable": true, "dtype": "float32", "function": "math.greater_equal"}, "name": "tf.math.greater_equal_10", "inbound_nodes": [["flatten_7", 0, 0, {"y": 0.0, "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_20", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_20", "inbound_nodes": [["embedding_23", 0, 0, {"y": ["embedding_21", 0, 0], "name": null}]]}, {"class_name": "Embedding", "config": {"name": "embedding_22", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 4, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding_22", "inbound_nodes": [[["tf.math.floormod_7", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.cast_10", "trainable": true, "dtype": "float32", "function": "cast"}, "name": "tf.cast_10", "inbound_nodes": [["tf.math.greater_equal_10", 0, 0, {"dtype": "float32"}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_21", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_21", "inbound_nodes": [["tf.__operators__.add_20", 0, 0, {"y": ["embedding_22", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.expand_dims_7", "trainable": true, "dtype": "float32", "function": "expand_dims"}, "name": "tf.expand_dims_7", "inbound_nodes": [["tf.cast_10", 0, 0, {"axis": -1}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.multiply_7", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "name": "tf.math.multiply_7", "inbound_nodes": [["tf.__operators__.add_21", 0, 0, {"y": ["tf.expand_dims_7", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.reduce_sum_7", "trainable": true, "dtype": "float32", "function": "math.reduce_sum"}, "name": "tf.math.reduce_sum_7", "inbound_nodes": [["tf.math.multiply_7", 0, 0, {"axis": 1}]]}], "input_layers": [["input_8", 0, 0]], "output_layers": [["tf.math.reduce_sum_7", 0, 0]]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 3]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 3]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "model_7", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_8"}, "name": "input_8", "inbound_nodes": []}, {"class_name": "Flatten", "config": {"name": "flatten_7", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_7", "inbound_nodes": [[["input_8", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.clip_by_value_10", "trainable": true, "dtype": "float32", "function": "clip_by_value"}, "name": "tf.clip_by_value_10", "inbound_nodes": [["flatten_7", 0, 0, {"clip_value_min": 0, "clip_value_max": 1000000.0}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.compat.v1.floor_div_7", "trainable": true, "dtype": "float32", "function": "compat.v1.floor_div"}, "name": "tf.compat.v1.floor_div_7", "inbound_nodes": [["tf.clip_by_value_10", 0, 0, {"y": 4, "name": null}]]}, {"class_name": "Embedding", "config": {"name": "embedding_23", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 52, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding_23", "inbound_nodes": [[["tf.clip_by_value_10", 0, 0, {}]]]}, {"class_name": "Embedding", "config": {"name": "embedding_21", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 13, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding_21", "inbound_nodes": [[["tf.compat.v1.floor_div_7", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.floormod_7", "trainable": true, "dtype": "float32", "function": "math.floormod"}, "name": "tf.math.floormod_7", "inbound_nodes": [["tf.clip_by_value_10", 0, 0, {"y": 4, "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.greater_equal_10", "trainable": true, "dtype": "float32", "function": "math.greater_equal"}, "name": "tf.math.greater_equal_10", "inbound_nodes": [["flatten_7", 0, 0, {"y": 0.0, "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_20", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_20", "inbound_nodes": [["embedding_23", 0, 0, {"y": ["embedding_21", 0, 0], "name": null}]]}, {"class_name": "Embedding", "config": {"name": "embedding_22", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 4, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding_22", "inbound_nodes": [[["tf.math.floormod_7", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.cast_10", "trainable": true, "dtype": "float32", "function": "cast"}, "name": "tf.cast_10", "inbound_nodes": [["tf.math.greater_equal_10", 0, 0, {"dtype": "float32"}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_21", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_21", "inbound_nodes": [["tf.__operators__.add_20", 0, 0, {"y": ["embedding_22", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.expand_dims_7", "trainable": true, "dtype": "float32", "function": "expand_dims"}, "name": "tf.expand_dims_7", "inbound_nodes": [["tf.cast_10", 0, 0, {"axis": -1}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.multiply_7", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "name": "tf.math.multiply_7", "inbound_nodes": [["tf.__operators__.add_21", 0, 0, {"y": ["tf.expand_dims_7", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.reduce_sum_7", "trainable": true, "dtype": "float32", "function": "math.reduce_sum"}, "name": "tf.math.reduce_sum_7", "inbound_nodes": [["tf.math.multiply_7", 0, 0, {"axis": 1}]]}], "input_layers": [["input_8", 0, 0]], "output_layers": [["tf.math.reduce_sum_7", 0, 0]]}}}
?
H	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.math.greater_equal_11", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.math.greater_equal_11", "trainable": true, "dtype": "float32", "function": "math.greater_equal"}}
?
I	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.concat_9", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.concat_9", "trainable": true, "dtype": "float32", "function": "concat"}}
?
J	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.clip_by_value_11", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.clip_by_value_11", "trainable": true, "dtype": "float32", "function": "clip_by_value"}}
?
K	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.cast_11", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.cast_11", "trainable": true, "dtype": "float32", "function": "cast"}}
?

Lkernel
Mbias
Ntrainable_variables
Oregularization_losses
P	variables
Q	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_27", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_27", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 512}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 512]}}
?
R	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.concat_10", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.concat_10", "trainable": true, "dtype": "float32", "function": "concat"}}
?

Skernel
Tbias
Utrainable_variables
Vregularization_losses
W	variables
X	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_28", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_28", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 256]}}
?

Ykernel
Zbias
[trainable_variables
\regularization_losses
]	variables
^	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_30", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_30", "trainable": true, "dtype": "float32", "units": 256, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 20}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 20]}}
?

_kernel
`bias
atrainable_variables
bregularization_losses
c	variables
d	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_29", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_29", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 256]}}
?

ekernel
fbias
gtrainable_variables
hregularization_losses
i	variables
j	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_31", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_31", "trainable": true, "dtype": "float32", "units": 256, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 256]}}
?
k	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.concat_11", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.concat_11", "trainable": true, "dtype": "float32", "function": "concat"}}
?

lkernel
mbias
ntrainable_variables
oregularization_losses
p	variables
q	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_32", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_32", "trainable": true, "dtype": "float32", "units": 256, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 512}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 512]}}
?
r	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.nn.relu_9", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.nn.relu_9", "trainable": true, "dtype": "float32", "function": "nn.relu"}}
?

skernel
tbias
utrainable_variables
vregularization_losses
w	variables
x	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_33", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_33", "trainable": true, "dtype": "float32", "units": 256, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 256]}}
?
y	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.__operators__.add_22", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.__operators__.add_22", "trainable": true, "dtype": "float32", "function": "__operators__.add"}}
?
z	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.nn.relu_10", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.nn.relu_10", "trainable": true, "dtype": "float32", "function": "nn.relu"}}
?

{kernel
|bias
}trainable_variables
~regularization_losses
	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_34", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_34", "trainable": true, "dtype": "float32", "units": 256, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 256]}}
?
?	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.__operators__.add_23", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.__operators__.add_23", "trainable": true, "dtype": "float32", "function": "__operators__.add"}}
?
?	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.nn.relu_11", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.nn.relu_11", "trainable": true, "dtype": "float32", "function": "nn.relu"}}
?
?	normalize
?trainable_variables
?regularization_losses
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Normalize", "name": "normalize_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "normalize_3", "trainable": true, "dtype": "float32"}}
?
?kernel
	?bias
?trainable_variables
?regularization_losses
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_35", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_35", "trainable": true, "dtype": "float32", "units": 4, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Constant", "config": {"value": 0}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 256]}}
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
 "
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
?
?layers
?metrics
 ?layer_regularization_losses
trainable_variables
?layer_metrics
regularization_losses
?non_trainable_variables
	variables
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "input_7", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 2]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 2]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_7"}}
?
?trainable_variables
?regularization_losses
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Flatten", "name": "flatten_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten_6", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
?
?	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.clip_by_value_9", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.clip_by_value_9", "trainable": true, "dtype": "float32", "function": "clip_by_value"}}
?
?	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.compat.v1.floor_div_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.compat.v1.floor_div_6", "trainable": true, "dtype": "float32", "function": "compat.v1.floor_div"}}
?
?
embeddings
?trainable_variables
?regularization_losses
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Embedding", "name": "embedding_20", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "embedding_20", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 52, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 2]}}
?
?
embeddings
?trainable_variables
?regularization_losses
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Embedding", "name": "embedding_18", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "embedding_18", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 13, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 2]}}
?
?	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.math.floormod_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.math.floormod_6", "trainable": true, "dtype": "float32", "function": "math.floormod"}}
?
?	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.math.greater_equal_9", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.math.greater_equal_9", "trainable": true, "dtype": "float32", "function": "math.greater_equal"}}
?
?	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.__operators__.add_18", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.__operators__.add_18", "trainable": true, "dtype": "float32", "function": "__operators__.add"}}
?
?
embeddings
?trainable_variables
?regularization_losses
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Embedding", "name": "embedding_19", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "embedding_19", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 4, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 2]}}
?
?	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.cast_9", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.cast_9", "trainable": true, "dtype": "float32", "function": "cast"}}
?
?	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.__operators__.add_19", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.__operators__.add_19", "trainable": true, "dtype": "float32", "function": "__operators__.add"}}
?
?	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.expand_dims_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.expand_dims_6", "trainable": true, "dtype": "float32", "function": "expand_dims"}}
?
?	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.math.multiply_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.math.multiply_6", "trainable": true, "dtype": "float32", "function": "math.multiply"}}
?
?	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.math.reduce_sum_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.math.reduce_sum_6", "trainable": true, "dtype": "float32", "function": "math.reduce_sum"}}
8
?0
?1
?2"
trackable_list_wrapper
 "
trackable_list_wrapper
8
?0
?1
?2"
trackable_list_wrapper
?
?layers
?metrics
 ?layer_regularization_losses
1trainable_variables
?layer_metrics
2regularization_losses
?non_trainable_variables
3	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "input_8", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 3]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_8"}}
?
?trainable_variables
?regularization_losses
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Flatten", "name": "flatten_7", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten_7", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
?
?	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.clip_by_value_10", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.clip_by_value_10", "trainable": true, "dtype": "float32", "function": "clip_by_value"}}
?
?	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.compat.v1.floor_div_7", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.compat.v1.floor_div_7", "trainable": true, "dtype": "float32", "function": "compat.v1.floor_div"}}
?
?
embeddings
?trainable_variables
?regularization_losses
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Embedding", "name": "embedding_23", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "embedding_23", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 52, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 3]}}
?
?
embeddings
?trainable_variables
?regularization_losses
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Embedding", "name": "embedding_21", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "embedding_21", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 13, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 3]}}
?
?	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.math.floormod_7", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.math.floormod_7", "trainable": true, "dtype": "float32", "function": "math.floormod"}}
?
?	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.math.greater_equal_10", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.math.greater_equal_10", "trainable": true, "dtype": "float32", "function": "math.greater_equal"}}
?
?	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.__operators__.add_20", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.__operators__.add_20", "trainable": true, "dtype": "float32", "function": "__operators__.add"}}
?
?
embeddings
?trainable_variables
?regularization_losses
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Embedding", "name": "embedding_22", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "embedding_22", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 4, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 3]}}
?
?	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.cast_10", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.cast_10", "trainable": true, "dtype": "float32", "function": "cast"}}
?
?	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.__operators__.add_21", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.__operators__.add_21", "trainable": true, "dtype": "float32", "function": "__operators__.add"}}
?
?	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.expand_dims_7", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.expand_dims_7", "trainable": true, "dtype": "float32", "function": "expand_dims"}}
?
?	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.math.multiply_7", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.math.multiply_7", "trainable": true, "dtype": "float32", "function": "math.multiply"}}
?
?	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.math.reduce_sum_7", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.math.reduce_sum_7", "trainable": true, "dtype": "float32", "function": "math.reduce_sum"}}
8
?0
?1
?2"
trackable_list_wrapper
 "
trackable_list_wrapper
8
?0
?1
?2"
trackable_list_wrapper
?
?layers
?metrics
 ?layer_regularization_losses
Dtrainable_variables
?layer_metrics
Eregularization_losses
?non_trainable_variables
F	variables
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
??2dense_27/kernel
:?2dense_27/bias
.
L0
M1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
L0
M1"
trackable_list_wrapper
?
?layers
?metrics
 ?layer_regularization_losses
Ntrainable_variables
?layer_metrics
Oregularization_losses
?non_trainable_variables
P	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
#:!
??2dense_28/kernel
:?2dense_28/bias
.
S0
T1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
S0
T1"
trackable_list_wrapper
?
?layers
?metrics
 ?layer_regularization_losses
Utrainable_variables
?layer_metrics
Vregularization_losses
?non_trainable_variables
W	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
": 	?2dense_30/kernel
:?2dense_30/bias
.
Y0
Z1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
Y0
Z1"
trackable_list_wrapper
?
?layers
?metrics
 ?layer_regularization_losses
[trainable_variables
?layer_metrics
\regularization_losses
?non_trainable_variables
]	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
#:!
??2dense_29/kernel
:?2dense_29/bias
.
_0
`1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
_0
`1"
trackable_list_wrapper
?
?layers
?metrics
 ?layer_regularization_losses
atrainable_variables
?layer_metrics
bregularization_losses
?non_trainable_variables
c	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
#:!
??2dense_31/kernel
:?2dense_31/bias
.
e0
f1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
e0
f1"
trackable_list_wrapper
?
?layers
?metrics
 ?layer_regularization_losses
gtrainable_variables
?layer_metrics
hregularization_losses
?non_trainable_variables
i	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
#:!
??2dense_32/kernel
:?2dense_32/bias
.
l0
m1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
l0
m1"
trackable_list_wrapper
?
?layers
?metrics
 ?layer_regularization_losses
ntrainable_variables
?layer_metrics
oregularization_losses
?non_trainable_variables
p	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
#:!
??2dense_33/kernel
:?2dense_33/bias
.
s0
t1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
s0
t1"
trackable_list_wrapper
?
?layers
?metrics
 ?layer_regularization_losses
utrainable_variables
?layer_metrics
vregularization_losses
?non_trainable_variables
w	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
#:!
??2dense_34/kernel
:?2dense_34/bias
.
{0
|1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
{0
|1"
trackable_list_wrapper
?
?layers
?metrics
 ?layer_regularization_losses
}trainable_variables
?layer_metrics
~regularization_losses
?non_trainable_variables
	variables
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
_tf_keras_layer?{"class_name": "Normalization", "name": "normalization_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "normalization_3", "trainable": true, "dtype": "float32", "axis": {"class_name": "__tuple__", "items": [-1]}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 256]}}
 "
trackable_list_wrapper
 "
trackable_list_wrapper
8
?0
?1
?2"
trackable_list_wrapper
?
?layers
?metrics
 ?layer_regularization_losses
?trainable_variables
?layer_metrics
?regularization_losses
?non_trainable_variables
?	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
": 	?2dense_35/kernel
:2dense_35/bias
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
?
?layers
?metrics
 ?layer_regularization_losses
?trainable_variables
?layer_metrics
?regularization_losses
?non_trainable_variables
?	variables
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
*:(	4?2embedding_20/embeddings
*:(	?2embedding_18/embeddings
*:(	?2embedding_19/embeddings
*:(	4?2embedding_23/embeddings
*:(	?2embedding_21/embeddings
*:(	?2embedding_22/embeddings
-:+?2 normalize_3/normalization_3/mean
1:/?2$normalize_3/normalization_3/variance
):'	 2!normalize_3/normalization_3/count
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
(
?0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
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
?metrics
 ?layer_regularization_losses
?trainable_variables
?layer_metrics
?regularization_losses
?non_trainable_variables
?	variables
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
 "
trackable_list_wrapper
(
?0"
trackable_list_wrapper
?
?layers
?metrics
 ?layer_regularization_losses
?trainable_variables
?layer_metrics
?regularization_losses
?non_trainable_variables
?	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
(
?0"
trackable_list_wrapper
 "
trackable_list_wrapper
(
?0"
trackable_list_wrapper
?
?layers
?metrics
 ?layer_regularization_losses
?trainable_variables
?layer_metrics
?regularization_losses
?non_trainable_variables
?	variables
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
 "
trackable_list_wrapper
(
?0"
trackable_list_wrapper
?
?layers
?metrics
 ?layer_regularization_losses
?trainable_variables
?layer_metrics
?regularization_losses
?non_trainable_variables
?	variables
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
?
?layers
?metrics
 ?layer_regularization_losses
?trainable_variables
?layer_metrics
?regularization_losses
?non_trainable_variables
?	variables
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
 "
trackable_list_wrapper
(
?0"
trackable_list_wrapper
?
?layers
?metrics
 ?layer_regularization_losses
?trainable_variables
?layer_metrics
?regularization_losses
?non_trainable_variables
?	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
(
?0"
trackable_list_wrapper
 "
trackable_list_wrapper
(
?0"
trackable_list_wrapper
?
?layers
?metrics
 ?layer_regularization_losses
?trainable_variables
?layer_metrics
?regularization_losses
?non_trainable_variables
?	variables
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
 "
trackable_list_wrapper
(
?0"
trackable_list_wrapper
?
?layers
?metrics
 ?layer_regularization_losses
?trainable_variables
?layer_metrics
?regularization_losses
?non_trainable_variables
?	variables
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
trackable_list_wrapper
 "
trackable_dict_wrapper
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
 "
trackable_dict_wrapper
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
??2Adam/dense_27/kernel/m
!:?2Adam/dense_27/bias/m
(:&
??2Adam/dense_28/kernel/m
!:?2Adam/dense_28/bias/m
':%	?2Adam/dense_30/kernel/m
!:?2Adam/dense_30/bias/m
(:&
??2Adam/dense_29/kernel/m
!:?2Adam/dense_29/bias/m
(:&
??2Adam/dense_31/kernel/m
!:?2Adam/dense_31/bias/m
(:&
??2Adam/dense_32/kernel/m
!:?2Adam/dense_32/bias/m
(:&
??2Adam/dense_33/kernel/m
!:?2Adam/dense_33/bias/m
(:&
??2Adam/dense_34/kernel/m
!:?2Adam/dense_34/bias/m
':%	?2Adam/dense_35/kernel/m
 :2Adam/dense_35/bias/m
/:-	4?2Adam/embedding_20/embeddings/m
/:-	?2Adam/embedding_18/embeddings/m
/:-	?2Adam/embedding_19/embeddings/m
/:-	4?2Adam/embedding_23/embeddings/m
/:-	?2Adam/embedding_21/embeddings/m
/:-	?2Adam/embedding_22/embeddings/m
(:&
??2Adam/dense_27/kernel/v
!:?2Adam/dense_27/bias/v
(:&
??2Adam/dense_28/kernel/v
!:?2Adam/dense_28/bias/v
':%	?2Adam/dense_30/kernel/v
!:?2Adam/dense_30/bias/v
(:&
??2Adam/dense_29/kernel/v
!:?2Adam/dense_29/bias/v
(:&
??2Adam/dense_31/kernel/v
!:?2Adam/dense_31/bias/v
(:&
??2Adam/dense_32/kernel/v
!:?2Adam/dense_32/bias/v
(:&
??2Adam/dense_33/kernel/v
!:?2Adam/dense_33/bias/v
(:&
??2Adam/dense_34/kernel/v
!:?2Adam/dense_34/bias/v
':%	?2Adam/dense_35/kernel/v
 :2Adam/dense_35/bias/v
/:-	4?2Adam/embedding_20/embeddings/v
/:-	?2Adam/embedding_18/embeddings/v
/:-	?2Adam/embedding_19/embeddings/v
/:-	4?2Adam/embedding_23/embeddings/v
/:-	?2Adam/embedding_21/embeddings/v
/:-	?2Adam/embedding_22/embeddings/v
?2?
#__inference__wrapped_model_20517371?
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
L__inference_custom_model_3_layer_call_and_return_conditional_losses_20518852
L__inference_custom_model_3_layer_call_and_return_conditional_losses_20519022
L__inference_custom_model_3_layer_call_and_return_conditional_losses_20518188
L__inference_custom_model_3_layer_call_and_return_conditional_losses_20518280?
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
1__inference_custom_model_3_layer_call_fn_20518603
1__inference_custom_model_3_layer_call_fn_20519091
1__inference_custom_model_3_layer_call_fn_20518442
1__inference_custom_model_3_layer_call_fn_20519160?
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
E__inference_model_6_layer_call_and_return_conditional_losses_20517474
E__inference_model_6_layer_call_and_return_conditional_losses_20519244
E__inference_model_6_layer_call_and_return_conditional_losses_20517506
E__inference_model_6_layer_call_and_return_conditional_losses_20519202?
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
*__inference_model_6_layer_call_fn_20517552
*__inference_model_6_layer_call_fn_20519257
*__inference_model_6_layer_call_fn_20517597
*__inference_model_6_layer_call_fn_20519270?
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
E__inference_model_7_layer_call_and_return_conditional_losses_20519312
E__inference_model_7_layer_call_and_return_conditional_losses_20517732
E__inference_model_7_layer_call_and_return_conditional_losses_20517700
E__inference_model_7_layer_call_and_return_conditional_losses_20519354?
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
*__inference_model_7_layer_call_fn_20517778
*__inference_model_7_layer_call_fn_20519367
*__inference_model_7_layer_call_fn_20519380
*__inference_model_7_layer_call_fn_20517823?
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
F__inference_dense_27_layer_call_and_return_conditional_losses_20519391?
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
+__inference_dense_27_layer_call_fn_20519400?
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
F__inference_dense_28_layer_call_and_return_conditional_losses_20519411?
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
+__inference_dense_28_layer_call_fn_20519420?
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
F__inference_dense_30_layer_call_and_return_conditional_losses_20519430?
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
+__inference_dense_30_layer_call_fn_20519439?
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
F__inference_dense_29_layer_call_and_return_conditional_losses_20519450?
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
+__inference_dense_29_layer_call_fn_20519459?
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
F__inference_dense_31_layer_call_and_return_conditional_losses_20519469?
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
+__inference_dense_31_layer_call_fn_20519478?
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
F__inference_dense_32_layer_call_and_return_conditional_losses_20519488?
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
+__inference_dense_32_layer_call_fn_20519497?
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
F__inference_dense_33_layer_call_and_return_conditional_losses_20519507?
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
+__inference_dense_33_layer_call_fn_20519516?
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
F__inference_dense_34_layer_call_and_return_conditional_losses_20519526?
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
+__inference_dense_34_layer_call_fn_20519535?
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
I__inference_normalize_3_layer_call_and_return_conditional_losses_20519552?
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
.__inference_normalize_3_layer_call_fn_20519561?
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
F__inference_dense_35_layer_call_and_return_conditional_losses_20519571?
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
+__inference_dense_35_layer_call_fn_20519580?
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
&__inference_signature_wrapper_20518682betscards0cards1"?
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
G__inference_flatten_6_layer_call_and_return_conditional_losses_20519586?
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
,__inference_flatten_6_layer_call_fn_20519591?
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
J__inference_embedding_20_layer_call_and_return_conditional_losses_20519601?
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
/__inference_embedding_20_layer_call_fn_20519608?
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
J__inference_embedding_18_layer_call_and_return_conditional_losses_20519618?
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
/__inference_embedding_18_layer_call_fn_20519625?
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
J__inference_embedding_19_layer_call_and_return_conditional_losses_20519635?
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
/__inference_embedding_19_layer_call_fn_20519642?
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
G__inference_flatten_7_layer_call_and_return_conditional_losses_20519648?
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
,__inference_flatten_7_layer_call_fn_20519653?
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
J__inference_embedding_23_layer_call_and_return_conditional_losses_20519663?
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
/__inference_embedding_23_layer_call_fn_20519670?
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
J__inference_embedding_21_layer_call_and_return_conditional_losses_20519680?
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
/__inference_embedding_21_layer_call_fn_20519687?
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
J__inference_embedding_22_layer_call_and_return_conditional_losses_20519697?
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
/__inference_embedding_22_layer_call_fn_20519704?
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
#__inference__wrapped_model_20517371?.???????????LMYZST_`eflmst{|????{?x
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
dense_35"?
dense_35??????????
L__inference_custom_model_3_layer_call_and_return_conditional_losses_20518188?.???????????LMYZST_`eflmst{|???????
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
L__inference_custom_model_3_layer_call_and_return_conditional_losses_20518280?.???????????LMYZST_`eflmst{|???????
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
L__inference_custom_model_3_layer_call_and_return_conditional_losses_20518852?.???????????LMYZST_`eflmst{|???????
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
L__inference_custom_model_3_layer_call_and_return_conditional_losses_20519022?.???????????LMYZST_`eflmst{|???????
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
1__inference_custom_model_3_layer_call_fn_20518442?.???????????LMYZST_`eflmst{|???????
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
1__inference_custom_model_3_layer_call_fn_20518603?.???????????LMYZST_`eflmst{|???????
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
1__inference_custom_model_3_layer_call_fn_20519091?.???????????LMYZST_`eflmst{|???????
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
1__inference_custom_model_3_layer_call_fn_20519160?.???????????LMYZST_`eflmst{|???????
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
F__inference_dense_27_layer_call_and_return_conditional_losses_20519391^LM0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? ?
+__inference_dense_27_layer_call_fn_20519400QLM0?-
&?#
!?
inputs??????????
? "????????????
F__inference_dense_28_layer_call_and_return_conditional_losses_20519411^ST0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? ?
+__inference_dense_28_layer_call_fn_20519420QST0?-
&?#
!?
inputs??????????
? "????????????
F__inference_dense_29_layer_call_and_return_conditional_losses_20519450^_`0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? ?
+__inference_dense_29_layer_call_fn_20519459Q_`0?-
&?#
!?
inputs??????????
? "????????????
F__inference_dense_30_layer_call_and_return_conditional_losses_20519430]YZ/?,
%?"
 ?
inputs?????????
? "&?#
?
0??????????
? 
+__inference_dense_30_layer_call_fn_20519439PYZ/?,
%?"
 ?
inputs?????????
? "????????????
F__inference_dense_31_layer_call_and_return_conditional_losses_20519469^ef0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? ?
+__inference_dense_31_layer_call_fn_20519478Qef0?-
&?#
!?
inputs??????????
? "????????????
F__inference_dense_32_layer_call_and_return_conditional_losses_20519488^lm0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? ?
+__inference_dense_32_layer_call_fn_20519497Qlm0?-
&?#
!?
inputs??????????
? "????????????
F__inference_dense_33_layer_call_and_return_conditional_losses_20519507^st0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? ?
+__inference_dense_33_layer_call_fn_20519516Qst0?-
&?#
!?
inputs??????????
? "????????????
F__inference_dense_34_layer_call_and_return_conditional_losses_20519526^{|0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? ?
+__inference_dense_34_layer_call_fn_20519535Q{|0?-
&?#
!?
inputs??????????
? "????????????
F__inference_dense_35_layer_call_and_return_conditional_losses_20519571_??0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????
? ?
+__inference_dense_35_layer_call_fn_20519580R??0?-
&?#
!?
inputs??????????
? "???????????
J__inference_embedding_18_layer_call_and_return_conditional_losses_20519618a?/?,
%?"
 ?
inputs?????????
? "*?'
 ?
0??????????
? ?
/__inference_embedding_18_layer_call_fn_20519625T?/?,
%?"
 ?
inputs?????????
? "????????????
J__inference_embedding_19_layer_call_and_return_conditional_losses_20519635a?/?,
%?"
 ?
inputs?????????
? "*?'
 ?
0??????????
? ?
/__inference_embedding_19_layer_call_fn_20519642T?/?,
%?"
 ?
inputs?????????
? "????????????
J__inference_embedding_20_layer_call_and_return_conditional_losses_20519601a?/?,
%?"
 ?
inputs?????????
? "*?'
 ?
0??????????
? ?
/__inference_embedding_20_layer_call_fn_20519608T?/?,
%?"
 ?
inputs?????????
? "????????????
J__inference_embedding_21_layer_call_and_return_conditional_losses_20519680a?/?,
%?"
 ?
inputs?????????
? "*?'
 ?
0??????????
? ?
/__inference_embedding_21_layer_call_fn_20519687T?/?,
%?"
 ?
inputs?????????
? "????????????
J__inference_embedding_22_layer_call_and_return_conditional_losses_20519697a?/?,
%?"
 ?
inputs?????????
? "*?'
 ?
0??????????
? ?
/__inference_embedding_22_layer_call_fn_20519704T?/?,
%?"
 ?
inputs?????????
? "????????????
J__inference_embedding_23_layer_call_and_return_conditional_losses_20519663a?/?,
%?"
 ?
inputs?????????
? "*?'
 ?
0??????????
? ?
/__inference_embedding_23_layer_call_fn_20519670T?/?,
%?"
 ?
inputs?????????
? "????????????
G__inference_flatten_6_layer_call_and_return_conditional_losses_20519586X/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? {
,__inference_flatten_6_layer_call_fn_20519591K/?,
%?"
 ?
inputs?????????
? "???????????
G__inference_flatten_7_layer_call_and_return_conditional_losses_20519648X/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? {
,__inference_flatten_7_layer_call_fn_20519653K/?,
%?"
 ?
inputs?????????
? "???????????
E__inference_model_6_layer_call_and_return_conditional_losses_20517474l????8?5
.?+
!?
input_7?????????
p

 
? "&?#
?
0??????????
? ?
E__inference_model_6_layer_call_and_return_conditional_losses_20517506l????8?5
.?+
!?
input_7?????????
p 

 
? "&?#
?
0??????????
? ?
E__inference_model_6_layer_call_and_return_conditional_losses_20519202k????7?4
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
E__inference_model_6_layer_call_and_return_conditional_losses_20519244k????7?4
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
*__inference_model_6_layer_call_fn_20517552_????8?5
.?+
!?
input_7?????????
p

 
? "????????????
*__inference_model_6_layer_call_fn_20517597_????8?5
.?+
!?
input_7?????????
p 

 
? "????????????
*__inference_model_6_layer_call_fn_20519257^????7?4
-?*
 ?
inputs?????????
p

 
? "????????????
*__inference_model_6_layer_call_fn_20519270^????7?4
-?*
 ?
inputs?????????
p 

 
? "????????????
E__inference_model_7_layer_call_and_return_conditional_losses_20517700l????8?5
.?+
!?
input_8?????????
p

 
? "&?#
?
0??????????
? ?
E__inference_model_7_layer_call_and_return_conditional_losses_20517732l????8?5
.?+
!?
input_8?????????
p 

 
? "&?#
?
0??????????
? ?
E__inference_model_7_layer_call_and_return_conditional_losses_20519312k????7?4
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
E__inference_model_7_layer_call_and_return_conditional_losses_20519354k????7?4
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
*__inference_model_7_layer_call_fn_20517778_????8?5
.?+
!?
input_8?????????
p

 
? "????????????
*__inference_model_7_layer_call_fn_20517823_????8?5
.?+
!?
input_8?????????
p 

 
? "????????????
*__inference_model_7_layer_call_fn_20519367^????7?4
-?*
 ?
inputs?????????
p

 
? "????????????
*__inference_model_7_layer_call_fn_20519380^????7?4
-?*
 ?
inputs?????????
p 

 
? "????????????
I__inference_normalize_3_layer_call_and_return_conditional_losses_20519552[??+?(
!?
?
x??????????
? "&?#
?
0??????????
? ?
.__inference_normalize_3_layer_call_fn_20519561N??+?(
!?
?
x??????????
? "????????????
&__inference_signature_wrapper_20518682?.???????????LMYZST_`eflmst{|???????
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
dense_35"?
dense_35?????????